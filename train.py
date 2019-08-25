import tensorflow as tf
from tensorflow.python.keras import callbacks
from config import config
import os
from model import DQSA
from logger_utils import get_logger
from Environment import OneTimeStepEnv
import numpy as np
from Memory import ExperienceReplay, Memory
from collections import deque
from random import randrange
import time
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


def createUserNets(users):
    userNets = []
    for usr in range(users):
        userNets.append(DQSA(input_size=config.input_size_user, usernet=True))
    return userNets


def initialize_history_input(users, env):
    history_input = deque(maxlen=config.usrTimeSteps)
    for tstep in range(config.usrTimeSteps):
        # for usr in range(users):
        #     rand_action = randrange(start=0, stop=config.Actions)
        #     env.step(action=rand_action, user=usr)
        # nextStateForEachUser, _ = env.getNextState()  # also resets the environment
        nextStateForEachUser = np.zeros((users, 1, 2 * config.K + 2)).astype(dtype=np.float32)
        history_input.append(nextStateForEachUser)
    return history_input


def synchWithCentral(userNets):
    """
    :param userNets:
    :return: userNets synched with central
    """
    for usr in userNets:
        usr.load_weights(config.ckpt_path)


def resetUserStates(userNets):
    """
    :param userNets:
    :return: usernets with resetted initial states for the lstm layer
    """
    for usrNet in userNets:
        usrNet.reset_states()

def getAction(Qvalues, temperature, alpha):
    """
    :param Qvalues:
    :param temperature:
    :param alpha:
    :return: an action according to equation 11 in the DQSA paper
    """
    softmaxValues = tf.nn.softmax(logits=-(Qvalues * temperature), axis=-1)
    action_dist = (1 - alpha) * softmaxValues + alpha * (1 / config.Actions) * (np.ones_like(softmaxValues))
    return np.squeeze(np.random.choice(config.Actions, p=np.squeeze(action_dist)))


def lower_epsilon(alpha):
    return max(0.00005, alpha * 0.9995)


def trainDqsa(callbacks, logger, centralNet:DQSA, centralTarget:DQSA):
    """
    :param callbacks: callbacks to save model and to write to TB
    :param logger: logger handler
    :param centralNet: the central net which we train every M episodes
    :param centralTarget: target central net for double DQN
    :return:
    """
    logger.info("start_training")
    Tensorcallback = callbacks['tensorboard']
    Checkpoint = callbacks['checkpoint']
    Checkpoint.on_epoch_end(epoch=0)
    env = OneTimeStepEnv()
    alpha = 0.95
    userNets = createUserNets(config.N)
    actionThatUsersChose = np.zeros((config.N, 1))
    ER = ExperienceReplay()
    synchWithCentral(userNets)   # synch with central
    for iteration in range(config.Iterations):
        # ----- start iteration loop -----
        if (iteration + 1) % 5 == 0:
            centralTarget.load_weights(config.ckpt_path)  # target synch with central
        logger.info("TargetNet synched")
        channelThroughPutMean = 0
        loss_value = 0
        collisonsMean = 0
        idle_timesMean = 0
        collisons = 0
        idle_times = 0
        channelThroughPut = 0
        for episode in range(config.Episodes):
            # ----- start episode loop -----
            episodeMemory = Memory(numOfUsers=config.N)  # initialize a memory for the episode
            if config.useUserHistory:
                history_input = initialize_history_input(config.N, env)
            resetUserStates(userNets)
            Xt = env.reset()
            Xt = np.expand_dims(Xt, axis=1)
            for tstep in range(config.TimeSlots):
                # ----- start time-steps loop -----
                if config.useUserHistory:  # using last states as input to the user, maybe he will learn better this way
                    history_input.append(np.squeeze(Xt, axis=1))
                    state = np.array(history_input)
                for usr in range(config.N):
                    if config.useUserHistory:
                        usr_state = np.expand_dims(np.squeeze(state[:, usr, :, :]), axis=0)
                    else:
                        usr_state = Xt[usr]
                    Qvalues = userNets[usr](usr_state)  # inserting the state at tstep to the DQSA
                    action = getAction(Qvalues=Qvalues, temperature=config.temperature_schedule(iteration), alpha=alpha)
                    actionThatUsersChose[usr] = action  # saving the action at time step tstep
                    env.step(action=action, user=usr)  # each user interact with the env by choosing an action
                # ----- end time-steps loop -----
                nextStateForEachUser, rewardForEachUser = env.getNextState()  # also resets the environment for the new tstep
                collisons += env.collisions
                idle_times += env.idle_times
                reward_sum = np.sum(rewardForEachUser)  # rewards are calculated by the ACK signal as competitive reward mechanism
                channelThroughPut += reward_sum
                episodeMemory.addTimeStepExperience(state=Xt, nextState=nextStateForEachUser,
                                                    rewards=rewardForEachUser, actions=np.squeeze(actionThatUsersChose))
                # accumulating the experience at time step tstep
                Xt = np.expand_dims(nextStateForEachUser, axis=1)  # state = next_State
            alpha = lower_epsilon(alpha)  # lowering the exploration rate
            ER.add_memory(memory=episodeMemory)  # after the tstep loop we accumulate the episode experience unto the ER
            if (episode + 1) % config.debug_freq == 0:  # for debugging purposes
                collisons /= config.TimeSlots * config.debug_freq
                idle_times /= config.TimeSlots * config.debug_freq
                channelThroughPut /= config.TimeSlots * config.debug_freq
                channelThroughPutMean += channelThroughPut
                collisonsMean += collisons
                idle_timesMean += idle_times
                tstlearningrate = centralNet.optimizer.learning_rate.numpy()
                logger.info(
                    "Iteration {}/{}- Episode {}/{}:  collisions {}, idle_times {} ,channelThroughput is {}, learning rate is {} and alpha is {}"
                    .format(iteration, config.Iterations, episode, config.Episodes, collisons,
                                idle_times, channelThroughPut, tstlearningrate, alpha))
                print(
                    "Iteration {}/{}- Episode {}/{}:  collisions {}, idle_times {}, channelThroughput is {}, learning rate is {} and alpha is {}"
                    .format(iteration, config.Iterations, episode, config.Episodes, collisons,
                                idle_times, channelThroughPut, tstlearningrate, alpha))
                collisons = 0
                idle_times = 0
                channelThroughPut = 0
            if ER.currentPosition % config.M == 0:  # training phase every M episodes we accumulate
                 loss = centralNet.fit(lr=config.learning_rate_schedule(iteration), centralTarget=centralTarget, ER=ER)
                 loss_value += loss
                 # logs = {'channelThroughput': channelThroughPut / config.TimeSlots}
                 Checkpoint.on_epoch_end(epoch=iteration)
                 ER.flush()  # clear out the ER after use
                 synchWithCentral(userNets)  # synch with central
        # ----- end episode loop -----
        channelThroughPutMean /= config.Episodes // config.debug_freq
        collisonsMean /= config.Episodes // config.debug_freq
        idle_timesMean /= config.Episodes // config.debug_freq
        logger.info("Iteration {}/{}: channelThroughput mean is {}, collisions is {} and idle_times {}"
                        .format(iteration, config.Iterations, channelThroughPutMean, collisonsMean, idle_timesMean))
        print("Iteration {}/{}:channelThroughput mean is {}, collisions is {} and idle_times {}"
                  .format(iteration, config.Iterations, channelThroughPutMean, collisonsMean, idle_timesMean ))
        logs = {'channelThroughput': channelThroughPutMean, "collisons": collisonsMean, "idle_times": idle_timesMean}
        Tensorcallback.on_epoch_end(epoch=iteration, logs=logs)
        # ----- end iteration loop -----


if __name__ == '__main__':

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_path):
            os.makedirs(config.model_path)
    optimizer = tf.compat.v2.optimizers.Adam()
    centralNet = DQSA(input_size=config.input_size_central, usernet=False)
    loss = tf.compat.v2.losses.mean_squared_error
    centralNet.compile(optimizer=optimizer, loss=loss)
    logger = get_logger(os.path.join(config.log_dir, "train_log"))
    Tensorcallback = callbacks.TensorBoard(config.log_dir,
                                           write_graph=False, write_images=False)
    Checkpoint = callbacks.ModelCheckpoint(filepath=config.model_path + "/checkpoint.hdf5", monitor='channelThroughput')
    Checkpoint.set_model(centralNet)
    Tensorcallback.set_model(centralNet)
    callbacks = {'tensorboard': Tensorcallback, 'checkpoint': Checkpoint}
    DQSATarget = DQSA(input_size=config.input_size_central, usernet=False)
    trainDqsa(callbacks, logger, centralNet, DQSATarget)
