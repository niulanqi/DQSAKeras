from Environment import Env, OneTimeStepEnv
import numpy as np
from config import config
from random import random, randrange
from utils.plot_utils import plot_graph

probabilityToTransmit = 1 / config.N


def slottedAlohaProtocol():
    rv = random()  # number between [0,1]
    if rv < probabilityToTransmit:  # Pr(Transmit = 1) = 1 / NumberOfUsers
        if config.K == 1:  # only one channel
            chosenAction = 1
        else:
            chosenAction = randrange(start=1, stop=config.K)
    else:
        chosenAction = 0
    return chosenAction


if __name__ == '__main__':
    env = OneTimeStepEnv()
    channelThroughPut = 0  # fraction of time that packets are successfully delivered over the channel
    # i.e no collisions or idle time slots
    for iteration in range(config.Iterations):
        _ = env.reset()
        for t in range(config.TimeSlots):
            env.reset()
            for user in range(config.N):
                action = slottedAlohaProtocol()
                env.step(action=action, user=user)
                # each user changes the inner state of the environment where the environment uses the inner state
                # in order to keep track on the channels and the ACK signals for each user
            nextStateForEachUser, rewardForEachUser = env.getNextState()
            # if a reward is one that means that a packets was successfully delivered over the channel
            # the sum has a maximum of the number of channels -> config.K
            channelThroughPut = channelThroughPut + np.sum(rewardForEachUser)
    # measuring the expected value
    channelThroughPut = channelThroughPut / (config.Iterations * config.TimeSlots)
    print("Channel Utilization average {}".format(channelThroughPut))
    ToPlotX = range(config.Iterations * config.TimeSlots)
    ToPlotY = np.ones_like(ToPlotX) * channelThroughPut
    plot_graph(data=[ToPlotX, ToPlotY], filename="Aloha", title="Aloha",
               xlabel="Time slot", ylabel="Average channel utilization", legend="SlottedAloha")


def testEnv():
    env = Env()
    channelThroughPut = 0  # fraction of time that packets are successfully delivered over the channel
    # i.e no collisions or idle time slots
    for iteration in range(config.Iterations):
        for t in range(config.TimeSlots):
            initialState = env.reset()
            for user in range(config.N):
                action = slottedAlohaProtocol()
                env.step(action=action, user=user)
                # each user changes the inner state of the environment where the environment uses the inner state
                # in order to keep track on the channels and the ACK signals for each user
            nextStateForEachUser, rewardForEachUser = env.getNextState()
            # if a reward is one that means that a packets was successfully delivered over the channel
            # the sum has a maximum of the number of channels -> config.K
            channelThroughPut = channelThroughPut + np.sum(rewardForEachUser)
    # measuring the expected value
    channelThroughPut = channelThroughPut / (config.Iterations * config.TimeSlots)
    print("Channel Utilization average {}".format(channelThroughPut))
    ToPlotX = range(config.Iterations * config.TimeSlots)
    ToPlotY = np.ones_like(ToPlotX) * channelThroughPut
    plot_graph(data=[ToPlotX, ToPlotY], filename="Aloha", title="Aloha",
               xlabel="Time slot", ylabel="Average channel utilization", legend="SlottedAloha")
#
#
# def testTimeEnv():
#     env = TimeDependentEnv()
#     channelThroughPut = 0  # fraction of time that packets are successfully delivered over the channel
#     # i.e no collisions or idle time slots
#     for iteration in range(config.Iterations):
#         TimeSPU = env.reset()
#         for t in range(config.TimeSlots):
#             env.resetTimeStep()
#             #  reset the internal state of the environment
#             #  which keep tracks of the users actions through out the time step
#             for user in range(config.N):
#                 action = slottedAlohaProtocol()
#                 env.step(action=action, user=user)
#                 # each user changes the inner state of the environment where the environment uses the inner state
#                 # in order to keep track on the channels and the ACK signals for each user
#             nextStateForEachUser, rewardForEachUser = env.tstep(timestep=t)
#             # if a reward is one that means that a packets was successfully delivered over the channel
#             # the sum has a maximum of the number of channels -> config.K
#             channelThroughPut = channelThroughPut + np.sum(rewardForEachUser)
#     # measuring the expected value
#     channelThroughPut = channelThroughPut / (config.Iterations * config.TimeSlots)
#     print("Channel Utilization average {}".format(channelThroughPut))
#     ToPlotX = range(config.Iterations * config.TimeSlots)
#     ToPlotY = np.ones_like(ToPlotX) * channelThroughPut
#     plot_graph(data=[ToPlotX, ToPlotY], filename="Aloha", title="Aloha",
#                xlabel="Time slot", ylabel="Average channel utilization", legend="SlottedAloha")