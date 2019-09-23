import tensorflow as tf
from tensorflow.python.keras import callbacks
from config import config
import os
from model import DQSAVersion2
from logger_utils import get_logger
from Environment import OneTimeStepEnv
import numpy as np
from Memory import ExperienceReplay, Memory
from collections import deque


def initCTP(time_slots=config.TimeSlots):
    channelThroughPutPerTstep = [[] for _ in range(time_slots + 2)]  # to restart each session we start and end
                                                                           # with zero
    channelThroughPutPerTstep[0].append(0)
    channelThroughPutPerTstep[-1].append(0)
    return channelThroughPutPerTstep


def getAction(Qvalues, temperature, alpha):
    """
    :param Qvalues:
    :param temperature:
    :param alpha:
    :return: an action according to equation 11 in the DQSA paper
    """
    softmaxValues = tf.nn.softmax(logits=(Qvalues - np.max(Qvalues) * np.ones_like(Qvalues)) * temperature)
    action_dist = (1 - alpha) * softmaxValues + alpha * (1 / config.Actions) * (np.ones_like(softmaxValues))
    action = np.squeeze(np.random.choice(config.Actions, p=np.squeeze(action_dist).ravel()))
    # action = np.argmax(action_dist)
    return action


def lower_epsilon(alpha):
    return max(0, alpha * 0.995)


def trainDqsa(callbacks, logger, centralNet:DQSAVersion2, centralTarget:DQSAVersion2):
    """
    training loop for the DQSA
    :param callbacks: callbacks to write to TB
    :param logger: logger handler
    :param centralNet: the central net which we train every M episodes
    :param centralTarget: target central net for double DQN
    """
    logger.info("start_training")
    Tensorcallback = callbacks['tensorboard']
    env = OneTimeStepEnv()
    alpha = 0.0
    beta = 20
    #userNets = createUserNets(config.N)
    userNet = DQSAVersion2(input_size=config.input_size_user, usernet=True)
    # synchWithCentral(userNets=userNets, path=config.load_ckpt_path)
    # actionThatUsersChose = np.zeros((config.N, 1))
    ER = ExperienceReplay()
    best_channel_throughput_so_far = 0
    channelThroughPutPerTstep = initCTP()  # init the data structure to view the mean reward at each t
    for iteration in range(config.Iterations):
        # ----- start iteration loop -----
        if (iteration + 1) % 2 == 0:
            if best_channel_throughput_so_far > 0.9:
                centralTarget.load_weights(config.best_ckpt_path)  # target synch with central
            else:
                centralTarget.load_weights(config.ckpt_path)
            logger.info("TargetNet synched")
        channelThroughPutMean = 0
        loss_value = []
        collisonsMean = 0
        idle_timesMean = 0
        collisons = 0
        idle_times = 0
        channelThroughPut = 0
        for episode in range(config.Episodes):
            # ----- start episode loop -----
            episodeMemory = Memory(numOfUsers=config.N)  # initialize a memory for the episode
            Xt = env.reset()
            userNet.reset_states()
            # Xt = np.expand_dims(Xt, axis=1)
            for tstep in range(config.TimeSlots):
                # ----- start time-steps loop -----
                UserQvalues = userNet(Xt)
                actionThatUsersChose = [getAction(Qvalues=UserQvalue, temperature=beta, alpha=alpha) for UserQvalue in UserQvalues]
                for usr in range(config.N):
                    env.step(action=actionThatUsersChose[usr], user=usr)
                nextStateForEachUser, rewardForEachUser, ack_vector = env.getNextState()  # also resets the env for the next t step
                episodeMemory.addTimeStepExperience(state=Xt, nextState=nextStateForEachUser,
                                                    rewards=rewardForEachUser, actions=np.squeeze(actionThatUsersChose))
                # accumulating the experience at time step tstep
                Xt = nextStateForEachUser  # state = next_State
                # for debug purposes
                collisons += env.collisions
                idle_times += env.idle_times
                ack_sum = np.sum(ack_vector)  # rewards are calculated by the ACK signal as competitive reward mechanism
                channelThroughPutPerTstep[tstep + 1].append(ack_sum)
                channelThroughPut += ack_sum
                # ----- end time-steps loop -----
            # the episode has ended so we add the episode memory to ER and reset the usr states
            ER.add_memory(memory=episodeMemory)  # after the tstep loop we insert the episode experience into the ER
            #resetUserStates(userNets)  # reset the user's lstm states
            if (episode + 1) % config.debug_freq == 0:  # for debugging purposes, please ignore
                collisons /= config.TimeSlots * config.debug_freq
                idle_times /= config.TimeSlots * config.debug_freq
                channelThroughPut /= config.TimeSlots * config.debug_freq
                channelThroughPutMean += channelThroughPut
                collisonsMean += collisons
                idle_timesMean += idle_times
                tstlearningrate = centralNet.model.optimizer.learning_rate.numpy()
                logger.info(
                    "Iteration {}/{}- Episode {}/{}:  collisions {}, idle_times {} ,channelThroughput is {}, learning rate is {}, beta is {} and alpha is {}"
                    .format(iteration, config.Iterations, episode, config.Episodes, collisons,
                                idle_times, channelThroughPut, tstlearningrate, beta, alpha))
                print(
                    "Iteration {}/{}- Episode {}/{}:  collisions {}, idle_times {}, channelThroughput is {}, learning rate is {}, beta is {} and alpha is {}"
                    .format(iteration, config.Iterations, episode, config.Episodes, collisons,
                                idle_times, channelThroughPut, tstlearningrate, beta, alpha))
                collisons = 0
                idle_times = 0
                channelThroughPut = 0
            if (ER.currentPosition) % config.M == 0:  # training phase every M episodes
                loss = centralNet.fit(lr=config.learning_rate_schedule(iteration) * config.learning_rate_multiplier, centralTarget=centralTarget, ER=ER)
                loss_value.append(loss)
                centralNet.save_weights(config.ckpt_path)  # save new weights (new policy) in ckpt_path
                ER.flush()  # clear out the ER after use
                if best_channel_throughput_so_far > 0.9:
                    userNet.load_weights(config.best_ckpt_path)  # target synch with central
                else:
                    userNet.load_weights(path=config.ckpt_path)
                # resetUserStates(userNets)  # reset the user's lstm states
        # ----- end episode loop -----
        channelThroughPutMean /= config.Episodes // config.debug_freq
        collisonsMean /= config.Episodes // config.debug_freq
        idle_timesMean /= config.Episodes // config.debug_freq
        loss_value = np.mean(loss_value)
        if (iteration + 1) % 5 == 0:
            # every debug freq iterations
            # we draw the mean reward for each time step
            channelThroughPutPerTstep = [np.mean(x) for x in channelThroughPutPerTstep]
            for i, x in enumerate(channelThroughPutPerTstep):
                logs = {'channelThroughputTstep': x}
                Tensorcallback.on_epoch_end(epoch=iteration * (config.TimeSlots + 2) + i, logs=logs)
            channelThroughPutPerTstep = initCTP()  # init the data structure
        #  every iteration we draw stuff on TB
        logger.info("Iteration{}/{}: channelThroughput mean is {}, loss {}, collisions is {} and idle_times {}"
                    .format(iteration, config.Iterations, channelThroughPutMean, loss_value, collisonsMean, idle_timesMean))
        print("Iteration  {}/{}:channelThroughput mean is {}, loss {} collisions is {} and idle_times {}"
              .format(iteration, config.Iterations, channelThroughPutMean, loss_value, collisonsMean, idle_timesMean ))
        logs = {'channelThroughput': channelThroughPutMean, "collisons": collisonsMean, "idle_times": idle_timesMean,
                "loss_value": loss_value}
        Tensorcallback.on_epoch_end(epoch=iteration, logs=logs)
        beta = config.temperature_schedule(beta=beta)
        alpha = lower_epsilon(alpha)  # lowering the exploration rate
        if best_channel_throughput_so_far < channelThroughPutMean:
            best_channel_throughput_so_far = channelThroughPutMean
            centralNet.save_weights(config.best_ckpt_path)
        # ----- end iteration loop -----


if __name__ == '__main__':
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_path):
            os.makedirs(config.model_path)
    optimizer = tf.compat.v2.optimizers.Adam()
    # centralNet = DQSA(input_size=config.input_size_central, usernet=False)
    loss = tf.compat.v2.losses.mean_squared_error
    centralNet = DQSAVersion2(input_size=config.input_size_central, usernet=False, optimizer=optimizer,
                              loss=loss)
    # centralNet.define_loss(loss=loss)
    # centralNet.define_optimizer(optimizer=optimizer)
    logger = get_logger(os.path.join(config.log_dir, "train_log"))
    Tensorcallback = callbacks.TensorBoard(config.log_dir,
                                           write_graph=True, write_images=False)
    Tensorcallback.set_model(centralNet.model)
    callbacks = {'tensorboard': Tensorcallback}
    DQSATarget = DQSAVersion2(input_size=config.input_size_central, usernet=False, optimizer=optimizer,
                              loss=loss)
    # centralNet.load_weights(path=config.load_ckpt_path)
    # DQSATarget.load_weights(path=config.load_ckpt_path)
    trainDqsa(callbacks, logger, centralNet, DQSATarget)
