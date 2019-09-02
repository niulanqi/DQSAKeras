import numpy as np
from config import config
from random import randrange
# ENUMS
NO_TRANSMISSION_SLOT = int(0)
TRANSMISSION = int(1)
NO_TRANSMISSION = int(0)


def competitive_reward_maximization(userVector):
    """

    :param userVector: a Vector of [Users, 2K + 2]
    :return: a vector of [Users,1] depicts the user's ACK signal
    """
    return userVector[:, -1]


class Env:

    def __init__(self, numOfChannels=config.K, numOfUsers=config.N):
        """
        initialize the Environment
        :param numOfChannels: Number of channels this Environment supports
        """
        #  vector that depicts the channels
        self.numOfChannels = numOfChannels
        self.numOfUsers = numOfUsers
        self.channels = np.zeros(numOfChannels + 1)
        self.capacities = np.ones(numOfChannels)
        self.ACK = [0]
        self.state = np.concatenate((self.channels, self.capacities, self.ACK))
        self.state = np.expand_dims(self.state, axis=0)
        # at the state vector each channel cell is described by the following:
        #   0 means idle slot
        #   1 means transmission
        #   > 1 means collison
        self.statePerUser = np.matmul(np.ones((1, self.numOfUsers)).T, self.state).copy().astype(dtype=np.float32)   # (Nx1) * (1 X (2K + 2)) -> (Nx(2K+2))
        # self.statePerUser[:, NO_TRANSMISSION_SLOT] = 1

    def reset(self):
        """
        Reset the Environment at each timestep
        :return: an initial state where all the users has yet to make a choice
        """
        # this commented out option is what i normally use, a randomized initial state
        # self.statePerUser = np.matmul(np.ones((1, self.numOfUsers)).T,
        #                     self.state).copy().astype(dtype=np.float32)  # (Nx1) * (1 X (2K + 2)) -> (Nx(2K+2))
        # self.statePerUser[:, NO_TRANSMISSION_SLOT] = 1
        # for usr in range(self.numOfUsers):
        #     action = randrange(start=0, stop=self.numOfChannels + 1)
        #     self.step(action=action, user=usr)
        # randomized_first_state = self.statePerUser.astype(dtype=np.float32)
        self.statePerUser = np.matmul(np.ones((1, self.numOfUsers)).T,
                                      self.state).copy().astype(dtype=np.float32)  # (Nx1) * (1 x (2K + 2)) -> Nx(2K+2)
        self.statePerUser[:, NO_TRANSMISSION_SLOT] = 1
        randomized_first_state = np.zeros_like(self.statePerUser)  # according to the matlab script
        randomized_first_state[:, self.numOfChannels + 1 : 2 * self.numOfChannels + 1] \
            = self.capacities + np.random.rand(*self.capacities.shape) * 1e-9
        return randomized_first_state

    # def step(self, action, user):
    #     """
    #     :param action: chosen channel or no transmission
    #     note: the Environment doesn't supply terminal state, that is because the terminal state of each time slot(!)
    #           is when all the users have chosen a course of action
    #     """
    #     assert 0 <= action <= self.numOfChannels
    #     if action != 0:
    #         self.statePerUser[user, NO_TRANSMISSION_SLOT] = 0  # the user chose to transmit
    #         self.statePerUser[user, action] = TRANSMISSION  # the user chose to transmit at channel "action"
    #         if np.sum(self.statePerUser[:, action]) <= TRANSMISSION:
    #             # Channel is been used by only one user there for there is No Collision
    #             self.statePerUser[user, -1] = TRANSMISSION  # ACK received
    #         else:  # Collision occurred
    #             indicesOfUsersThatChoseTheSameChannel = self.statePerUser[:, action] == TRANSMISSION
    #             indicesOfUsersThatChoseTheSameChannel = indicesOfUsersThatChoseTheSameChannel.astype(np.int)
    #             # get all the users that transmitted on that channel and turn their ACK signal to 0
    #             indicesOfUsersThatChoseTheSameChannel = np.argwhere(indicesOfUsersThatChoseTheSameChannel)
    #             self.statePerUser[indicesOfUsersThatChoseTheSameChannel, -1] = NO_TRANSMISSION
    #     else:  # action means no transmission
    #         self.statePerUser[user, NO_TRANSMISSION_SLOT] = 1
    #         self.statePerUser[user, -1] = 0  # ACK signal is 0 when not transmitting
    def step(self, action, user):
        """
        :param action: chosen channel or no transmission
        note: the Environment doesn't supply terminal state, that is because the terminal state of each time slot(!)
              is when all the users have chosen a course of action
        """
        assert 0 <= action <= self.numOfChannels
        if action != 0:
            self.statePerUser[user, NO_TRANSMISSION_SLOT] = 0  # the user chose to transmit
            self.statePerUser[user, action] = TRANSMISSION  # the user chose to transmit at channel "action"
            if np.sum(self.statePerUser[:, action]) <= TRANSMISSION:
                # Channel is been used by only one user there for there is No Collision
                self.statePerUser[user, -1] = TRANSMISSION  # ACK received
                #self.statePerUser[:, self.numOfChannels + action] = 0
                # The channel is being used so the capacity is zero
            else:  # Collision occurred
                indicesOfUsersThatChoseTheSameChannel = self.statePerUser[:, action] == TRANSMISSION
                indicesOfUsersThatChoseTheSameChannel = indicesOfUsersThatChoseTheSameChannel.astype(np.int)
                # get all the users that transmitted on that channel and turn their ACK signal to 0
                indicesOfUsersThatChoseTheSameChannel = np.argwhere(indicesOfUsersThatChoseTheSameChannel)
                #self.statePerUser[:, self.numOfChannels + action] = 1
                # the channel is not being used due to a collison so the capacity is one
                self.statePerUser[indicesOfUsersThatChoseTheSameChannel, -1] = NO_TRANSMISSION  # ACK is zero

        else:  # action means no transmission
            self.statePerUser[user, NO_TRANSMISSION_SLOT] = 1
            self.statePerUser[user, -1] = 0  # ACK signal is 0 when not transmitting

    def getNextState(self):
        """
        function to return a next state and reward for each user at the end of the time slot after all the users
        have chosen a channel
        :return: a Vector of [Users, 2K + 3] where the first 2K + 2 elements are the next state and the last element is the reward
        """
        reward_vector = competitive_reward_maximization(self.statePerUser)
        collison_flag = np.sum(self.statePerUser[:, 1])
        self.collisions = collison_flag if collison_flag > 1 else 0
        self.idle_times = np.sum(self.statePerUser[:, NO_TRANSMISSION_SLOT])
        return self.statePerUser.astype(dtype=np.float32), reward_vector


class OneTimeStepEnv(Env):

    def reset(self):
        state = super(OneTimeStepEnv, self).reset()
        return np.expand_dims(state, axis=1).copy()
        # adding the time step dimension in the first axis ( only when Time major is off(!!!!))

    def getNextState(self):
        nextState, reward_vector = super(OneTimeStepEnv, self).getNextState()
        self.reset()
        return np.expand_dims(nextState, axis=1).copy(), reward_vector

#
#
# class TimeDependentEnv(Env):
#     """
#     an environment for the DQSA algorithm, this will use the Env class to create observations  at each time step
#     and will aggregate them through the time step dimension that enters the DQSA architecture
#     """
#     def __init__(self, numOfChannels=config.K, numOfUsers=config.N, TimeSlots=config.TimeSlotsForLstm):
#         super(TimeDependentEnv, self).__init__(numOfChannels=numOfChannels, numOfUsers=numOfUsers)
#         self.Timeslots = TimeSlots
#         self.TimeSPU = np.zeros((TimeSlots, numOfUsers, 2 * numOfChannels + 2))
#         self.TimeSPU[range(self.Timeslots), :, :] = self.statePerUser
#
#     def resetTimeStep(self):
#         super(TimeDependentEnv, self).reset()
#
#     def reset(self):
#         """
#         this function reset the environment at each episode, it return a TimeSPU that is initialized
#         throughout the Time dimension
#         :return: a clean TimeSPU for a new episode
#         """
#         super(TimeDependentEnv, self).reset()
#         self.TimeSPU[range(self.Timeslots), :, :] = self.statePerUser
#         return self.TimeSPU
#
#     def tstep(self, timestep):
#         """
#         signals the class that a time step has been completed and that all users submitted their actions
#         that means the statePerUser (or SPU in short) filled with the users actions for the last time step
#         :return:
#         """
#         # creates the next state of the environment which is the aggregated observation through time
#         self.TimeSPU[timestep, :, :] = self.statePerUser
#         #  receive the reward vector for all the users for their actions in  time step
#         _, reward_vector = self.getNextState()
#         return self.TimeSPU, reward_vector
#
#
