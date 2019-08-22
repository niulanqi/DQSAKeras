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
        for usr in range(users):
            rand_action = randrange(start=0, stop=config.Actions)
            env.step(action=rand_action, user=usr)
        nextStateForEachUser, _ = env.getNextState()  # also resets the environment
        history_input.append(nextStateForEachUser)
    return history_input


def synchWithCentral(userNets):
    for usr in userNets:
        usr.load_weights(config.ckpt_path)


def  resetUserStates(userNets):
    for usrNet in userNets:
        usrNet.reset_states()

def getAction(Qvalues, temperature, alpha):
    softmaxValues = tf.nn.softmax(logits=-(Qvalues * temperature), axis=-1)
    action_dist = (1 - alpha) * softmaxValues + alpha * (1 / config.Actions) * (np.ones_like(softmaxValues))
    return np.squeeze(np.random.choice(config.Actions, p=np.squeeze(action_dist)))


def lower_epsilon(alpha):
    return max(0.00005, alpha * 0.9995)


def trainDqsa(callbacks, logger, centralNet:DQSA, centralTarget:DQSA):
    logger.info("start_training")
    Tensorcallback = callbacks['tensorboard']
    Checkpoint = callbacks['checkpoint']
    channelThroughPutMean = 0
    collisons = 0
    idle_times = 0
    logs = {'channelThroughput': channelThroughPutMean, "collisons": collisons, "idle_times": idle_times}
    Tensorcallback.on_epoch_end(epoch=0, logs=logs)
    Checkpoint.on_epoch_end(epoch=0)
    env = OneTimeStepEnv()
    alpha = 0.9
    userNets = createUserNets(config.N)
    actionThatUsersChose = np.zeros((config.N, 1))
    ER = ExperienceReplay()
    synchWithCentral(userNets)
    for iteration in range(config.Iterations):
        if (iteration + 1) % 5 == 0:
            centralTarget.load_weights(config.ckpt_path)
        logger.info("TargetNet synched")
        channelThroughPutMean = 0
        loss_value = 0
        collisonsMean = 0
        idle_timesMean = 0
        collisons = 0
        idle_times = 0
        channelThroughPut = 0
        for episode in range(config.Episodes):
            episodeMemory = Memory(numOfUsers=config.N)
            #history_input = initialize_history_input(config.N, env)
            resetUserStates(userNets)
            Xt = env.reset()
            Xt = np.expand_dims(Xt, axis=1)
            for tstep in range(config.TimeSlots):
                #history_input.append(np.squeeze(Xt, axis=1))
                #state = np.array(history_input)
                for usr in range(config.N):
                    #usr_state = np.expand_dims(np.squeeze(state[:, usr, :, :]), axis=0)
                    # tick = time.time()
                    #Qvalues = userNets[usr](usr_state)
                    Qvalues = userNets[usr](Xt[usr])
                    # tock = time.time()
                    # print("time: {}".format(tock - tick))
                    action = getAction(Qvalues=Qvalues, temperature=config.temperature_schedule(iteration), alpha=alpha)
                    actionThatUsersChose[usr] = action
                    env.step(action=action, user=usr)
                nextStateForEachUser, rewardForEachUser = env.getNextState()  # also resets the environmen
                collisons += env.collisions
                idle_times += env.idle_times
                reward_sum = np.sum(rewardForEachUser)
                # rewards_per_timestep[tstep].append(reward_sum)
                channelThroughPut += reward_sum
                episodeMemory.addTimeStepExperience(state=Xt, nextState=nextStateForEachUser,
                                                    rewards=rewardForEachUser, actions=np.squeeze(actionThatUsersChose))
                Xt = np.expand_dims(nextStateForEachUser, axis=1)  # state = next_State
            alpha = lower_epsilon(alpha)

            ER.add_memory(memory=episodeMemory)
            if (episode + 1) % 10 == 0:
                collisons /= config.TimeSlots * 10
                idle_times /= config.TimeSlots * 10
                channelThroughPut /= config.TimeSlots * 10
                channelThroughPutMean += channelThroughPut
                collisonsMean += collisons
                idle_timesMean += idle_times
                logger.info(
                    "Iteration {}/{}- Episode {}/{}:  collisions {}, idle_times {} ,channelThroughput is {} and alpha is {}"
                    .format(iteration, config.Iterations, episode, config.Episodes, collisons,
                                idle_times, channelThroughPut, alpha))
                print(
                    "Iteration {}/{}- Episode {}/{}:  collisions {}, idle_times {}, channelThroughput is {} and alpha is {}"
                    .format(iteration, config.Iterations, episode, config.Episodes, collisons,
                                idle_times, channelThroughPut, alpha))
                collisons = 0
                idle_times = 0
                channelThroughPut = 0
            if ER.currentPosition % config.M == 0:
                 loss = centralNet.fit(lr=config.learning_rate_schedule(iteration), centralTarget=centralTarget, ER=ER)
                 loss_value += loss
                 # logs = {'channelThroughput': channelThroughPut / config.TimeSlots}
                 Checkpoint.on_epoch_end(epoch=iteration)
                 ER.flush()
                 synchWithCentral(userNets)

        channelThroughPutMean /= config.Episodes // 10
        logger.info("Iteration {}/{}: channelThroughput mean is {}, collisions is {} and idle_times {}"
                        .format(iteration, config.Iterations, channelThroughPutMean, collisonsMean / config.Episodes, idle_timesMean / config.Episodes))
        print("Iteration {}/{}:channelThroughput mean is {}, collisions is {} and idle_times {}"
                  .format(iteration, config.Iterations, channelThroughPutMean, collisonsMean / config.Episodes, idle_timesMean / config.Episodes))
        logs = {'channelThroughput': channelThroughPutMean, "collisons": collisonsMean / config.Episodes, "idle_times": idle_timesMean / config.Episodes}
        Tensorcallback.on_epoch_end(epoch=iteration, logs=logs)


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
