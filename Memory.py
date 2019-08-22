import random
from collections import namedtuple
from config import config
import numpy as np

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state'])
SequenceExperience = namedtuple('SequenceExperience', ['state', 'action', 'reward', 'next_state'])
#
#
# class ExperienceReplay:
#     """
#     This class provides an abstraction to store the [s, a, r, s'] elements of each iteration.
#     using Experience object which contains the s-a-r-s-a transition information in an object oriented way
#     """
#
#     def __init__(self, size):
#             self.size = size
#             self.currentPosition = 0
#             self.buffer = []
#
#     def add_memory(self, state, action, reward, next_state, terminal):
#         exp = Experience(state, action, reward, next_state, terminal)
#         if len(self.buffer) < self.size:
#             self.buffer.append(exp)
#         else:
#             self.buffer[self.currentPosition] = exp
#         self.currentPosition = (self.currentPosition + 1) % self.size
#
#     def getMiniBatch(self, batch_size):
#         indices = random.sample(population=range(len(self.buffer)), k=min(batch_size, len(self.buffer)))
#         return [self.buffer[index] for index in indices]

# class UserMemory:
#     """
#     This class provides a data structure that houses the cell state and hidden state of the user from the last time step
#     """
#     def __init__(self, numOfUsers=config.N, hidden_state_and_cell_size=config.LstmUnits):
#         self.userMemory = np.zeros((numOfUsers, 2, hidden_state_and_cell_size))
#
#     def add_memory(self, user, cell_state, hidden_state):
#             self.userMemory[user, 0, :] = cell_state
#             self.userMemory[user, 1, :] = hidden_state
#
#     def getUserMemory(self, user):
#         """
#         retrieve the last hidden state and cell state of user
#         :param user: to indiciate the user requesting its last hidden state and memory
#         :return: previous hidden cell state and hidden state
#         """
#
#         return self.userMemory[user, :, 0, :],  self.userMemory[user, :, 1, :]
#
#     def initializeMemory(self):
#         self.userMemory = np.zeros_like(self.userMemory)
#



def getSequence(timeExp:SequenceExperience , seq_length: int):
    assert len(timeExp.state) > seq_length
    return SequenceExperience(state=timeExp.state[:seq_length], next_state=timeExp.next_state[:seq_length],
                              action=timeExp.action[:seq_length], reward=timeExp.reward[:seq_length] )

class Memory:
    def __init__(self, numOfUsers=config.N):
        self.numUsers = numOfUsers
        self.states = [[] for _ in range(numOfUsers)]
        self.next_states = [[] for _ in range(numOfUsers)]
        self.reward = [[] for _ in range(numOfUsers)]
        self.action = [[] for _ in range(numOfUsers)]


    def addTimeStepExperience(self, state, nextState, rewards, actions):
        for usr in range(self.numUsers):
                self.states[usr].append(state[usr])
                self.next_states[usr].append(nextState[usr])
                self.reward[usr].append(rewards[usr])
                self.action[usr].append(actions[usr])


    def withdrawSequenceOfExperiences(self, seq_length: int):  #TODO further enhancement will be to samle the sequence not only from the start
        states = [per_user[:seq_length] for per_user in self.states]
        next_states = [per_user[:seq_length] for per_user in self.next_states]
        reward = [per_user[:seq_length] for per_user in self.reward]
        action = [per_user[:seq_length] for per_user in self.action]
        return SequenceExperience(state=states, action=action, reward=reward, next_state=next_states)


class ExperienceReplay:
    """
    This class provides an abstraction to store Memories and withdraw random batches of them
    """

    def __init__(self, size=config.memory_size):
            self.size = size
            self.currentPosition = 0
            self.buffer = []

    def add_memory(self, memory: Memory):
        if len(self.buffer) < self.size:
            self.buffer.append(memory)
        else:
            self.buffer[self.currentPosition] = memory
        self.currentPosition = (self.currentPosition + 1) % self.size

    def getMiniBatch(self, seq_length,  batch_size=config.batch_size):
        indices = random.sample(population=range(len(self.buffer)), k=min(batch_size, len(self.buffer)))
        return [self.buffer[index].withdrawSequenceOfExperiences(seq_length=seq_length) for index in indices]

    def flush(self):
        """
        should flush the experiences each training session
        """
        self.buffer = []
        self.currentPosition = 0

