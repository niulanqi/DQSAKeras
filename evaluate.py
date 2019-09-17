import tensorflow as tf
from tensorflow.python.keras import callbacks
from config import config
import os
from model import DQSA, DQSAVersion2
from logger_utils import get_logger
from Environment import OneTimeStepEnv
import numpy as np
from Memory import ExperienceReplay, Memory
from collections import deque
from train import createUserNets, initCTP, getAction, resetUserStates, synchWithCentral


if __name__ == '__main__':
    userNets = createUserNets(config.N)
    synchWithCentral(userNets, path=config.load_ckpt_path)
    logger = get_logger(os.path.join(config.evaluate_log_dir, "evaluate_log"))
    Tensorcallback = callbacks.TensorBoard(config.evaluate_log_dir,
                                           write_graph=True, write_images=False)
    Tensorcallback.set_model(userNets[0].model)
    env = OneTimeStepEnv()
    beta = 10
    alpha = 0  # e_greedy
    channelThroughPutPerTstep = initCTP(config.TimeSlotsEvaluate)  # init the data structure to view the mean reward at each t
    for iteration in range(2):
        channelThroughPutMean = 0
        loss_value = []
        collisonsMean = 0
        idle_timesMean = 0
        for episode in range(config.Episodes):
            collisons = 0
            idle_times = 0
            channelThroughPut = 0
            Xt = env.reset()
            # Xt = np.expand_dims(Xt, axis=1)
            for tstep in range(config.TimeSlotsEvaluate):
                for usr in range(config.N):  # each usr interacts with the env in this loo
                    usr_state = Xt[usr]
                    Qvalues = userNets[usr](usr_state)  # inserting the state to the stateful DQSA
                    action = getAction(Qvalues=np.squeeze(Qvalues), temperature=beta, alpha=alpha)
                    env.step(action=action, user=usr)  # each user interact with the env by choosing an action
                nextStateForEachUser, rewardForEachUser = env.getNextState()  # also resets the env for the next t step
                Xt = np.expand_dims(nextStateForEachUser, axis=1)  # state = next_State
                collisons += env.collisions
                idle_times += env.idle_times
                reward_sum = np.sum(
                    rewardForEachUser)  # rewards are calculated by the ACK signal as competitive reward mechanism
                tmp = channelThroughPutPerTstep[tstep + 1]
                channelThroughPutPerTstep[tstep + 1].append(reward_sum)
                channelThroughPut += reward_sum
            resetUserStates(userNets)  # reset the user's lstm states
            collisonsMean += collisons / config.TimeSlotsEvaluate
            idle_timesMean += idle_times / config.TimeSlotsEvaluate
            channelThroughPutMean += channelThroughPut / config.TimeSlotsEvaluate
            print(
                "Iteration {}/{}- Episode {}/{}:  collisions {}, idle_times {}, channelThroughput is {}"
                    .format(iteration, 10, episode, config.Episodes, collisons / config.TimeSlotsEvaluate,
                            idle_times / config.TimeSlotsEvaluate, channelThroughPut / config.TimeSlotsEvaluate))
            logger.info(
                "Iteration {}/{}- Episode {}/{}:  collisions {}, idle_times {}, channelThroughput is {}"
                    .format(iteration, 10, episode, config.Episodes, collisons / config.TimeSlotsEvaluate,
                            idle_times / config.TimeSlotsEvaluate, channelThroughPut / config.TimeSlotsEvaluate))

        channelThroughPutPerTstep = [np.mean(x) for x in channelThroughPutPerTstep]
        for i, x in enumerate(channelThroughPutPerTstep):
            logs = {'channelThroughputTstep': x}
            Tensorcallback.on_epoch_end(epoch=iteration * (config.TimeSlotsEvaluate + 2) + i, logs=logs)
        channelThroughPutPerTstep = initCTP(config.TimeSlotsEvaluate)  # init the data structure
        print(
            "Iteration {}/{}  collisions {}, idle_times {}, channelThroughput is {}"
                .format(iteration, 10, collisonsMean / config.Episodes,
                        idle_timesMean / config.Episodes, channelThroughPutMean / config.Episodes))
        logger.info(
            "Iteration {}/{}  collisions {}, idle_times {}, channelThroughput is {}"
                .format(iteration, 10, collisonsMean / config.Episodes,
                        idle_timesMean / config.Episodes, channelThroughPutMean / config.Episodes))
