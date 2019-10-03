import tensorflow as tf
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.activations import *
from config import config
from Memory import ExperienceReplay
import numpy as np
# from random import randrange
# from memory_profiler import profile

from PeepholesLSTM import PeepholesLSTM


def processing_experience(exp_batch):
    states_processed = []
    actions_processed = []
    next_states_processed = []
    rewards_processed = []
    for exp in exp_batch:
        states_processed.extend(exp.state)
        actions_processed.extend(exp.action)
        next_states_processed.extend(exp.next_state)
        rewards_processed.extend(exp.reward)
    states_processed = np.squeeze(np.asarray(states_processed))
    actions_processed = np.squeeze(np.asarray(actions_processed))
    next_states_processed = np.squeeze(np.asarray(next_states_processed))
    rewards_processed = np.squeeze(np.asarray(rewards_processed))
    next_states_processed = np.concatenate((np.expand_dims(states_processed[:, 0, :], axis=1), next_states_processed),
                                           axis=1)
    return states_processed[:12, :, :], actions_processed[:12, :], next_states_processed[:12, :, :], rewards_processed[:12, :]


def dqsa(usernet, input_size, batch_size):
    input_layer = Input(shape=input_size[1:], batch_size=batch_size)
    lstm_layer = PeepholesLSTM(units=config.LstmUnits, stateful=usernet, return_sequences=True)(input_layer) # allows better timing
    streamAC = TimeDistributed(Dense(units=10, activation=tanh), input_shape=(input_size[1], config.LstmUnits))(lstm_layer)
    streamVC = TimeDistributed(Dense(units=10, activation=tanh), input_shape=(input_size[1], config.LstmUnits))(lstm_layer)
    advantage = TimeDistributed(Dense(units=config.Actions))(streamAC)
    value = TimeDistributed(Dense(units=1))(streamVC)
    pred = value + advantage - tf.reduce_mean(advantage)
    model = Model(inputs=input_layer, outputs=pred)
    return model


# class DQSA(tf.keras.Model):
#     def __init__(self, input_size, usernet, batch_size=None):
#         """
#         creating an instance of a DQSA model
#         :param input_size: input size of the network
#         :param usernet: a signal to determine if the network should be stateful or not
#                (usernet = stateful)
#         """
#         self.batch_size = batch_size
#         super(DQSA, self).__init__()
#         self.lstm = PeepholesLSTM(units=config.LstmUnits, stateful=usernet, return_sequences=True, batch_size=batch_size)
#         self.streamAC = TimeDistributed(Dense(units=10, activation=tanh), input_shape=(input_size[1], config.LstmUnits))
#         self.streamVC = TimeDistributed(Dense(units=10, activation=tanh), input_shape=(input_size[1], config.LstmUnits))
#         self.advantage = TimeDistributed(Dense(units=config.Actions))
#         self.value = TimeDistributed((Dense(units=1)))
#         self.pred = Add()
#         # according to the matlab script we need to get rid of the biases for the deterministic values
#
#     @tf.function
#     def call(self, inputs):
#         """
#         forward pass
#         :param inputs: input states to the DQSA
#         :return: prediction of shape K + 1
#         """
#         # pred = self.model(inputs)
#         lstmOutput = self.lstm(inputs)
#         streamVC = self.streamVC(lstmOutput)
#         streamAC = self.streamAC(lstmOutput)
#         advantage = self.advantage(streamAC)
#         value = self.value(streamVC)
#         pred = self.pred([value, advantage - tf.reduce_mean(advantage)])
#         return pred
#
#     def define_optimizer(self, optimizer):
#         self.optimizer = optimizer
#
#     def define_loss(self, loss):
#         self.loss = loss
#
#     def get_batch_size(self):
#         return self.batch_size
#
#
#     @profile
#     def fit(self, lr, ER: ExperienceReplay, centralTarget):
#         """
#         fitting the model, the current version evaluates the target vector for every time step (while taking into
#         consideration the sequence leading to that time step), the target vector is evaluated using the latest policy
#         once we evaluate the entire target vector, we apply backprop with that target vector
#         :param lr: learning rate
#         :param ER: Experience Replay
#         :param centralTarget: central target
#         :return: mean loss value
#         """
#         lossValue = []
#         self.optimizer.learning_rate = lr  # deciding the optimizer learning rate
#         for _ in range(config.train_iterations):
#             seq_len = config.TimeSlots  # following the paper
#             #  seq_len = randrange(start=5, stop=config.TimeSlots)
#             exp_batch = ER.getMiniBatch(batch_size=config.batch_size, seq_length=seq_len)
#             states_processed, actions_processed, next_states_processed, rewards_processed = processing_experience(exp_batch)
#             # done organizing data to shape (number_of_users * batch size , 50, 2K + 2)
#             target_vector = self(states_processed)
#             target_vector = target_vector.numpy()
#             next_state_qvalues = self(next_states_processed)
#             next_state_qvalues = next_state_qvalues[:, 1:, :].numpy()
#             next_state_qvalues_target = centralTarget(next_states_processed)
#             next_state_qvalues_target = next_state_qvalues_target[:, 1:, :].numpy()
#             evaluated_actions = np.argmax(next_state_qvalues, axis=-1).astype(np.int32)
#             for t in range(seq_len):
#                 double_dqn = np.asarray([Qvalue[evaluated_actions[i, t]] for i, Qvalue in enumerate(next_state_qvalues_target[:, t, :])])
#                 for i, nextQvalue in enumerate(double_dqn):  # creating the labels
#                     target_vector[i, t, int(actions_processed[i, t])] = rewards_processed[i, t] + config.Gamma * nextQvalue
#             loss, grads = self.trainPhase(states=states_processed, target_vector=target_vector)
#             self.applyGradients(grads=grads)
#             # loss = self.train_on_batch(x=states_processed, y=target_vector)
#             lossValue.append(loss)
#         return np.mean(lossValue)
# #
#     @tf.function
#     def trainPhase(self, target_vector, states):
#         with tf.GradientTape() as tape:
#             current_predictions = self(states)
#             train_loss = self.loss(target_vector, current_predictions)
#         grads = tape.gradient(train_loss, self.trainable_variables)
#         return train_loss, grads
#
#     @tf.function
#     def applyGradients(self, grads):
#         self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
# # #
# #     @tf.function
# #     def add(self, a, b):
# #         return tf.add(a, b)


class DQSAVersion2:
    def __init__(self, input_size, usernet, batch_size, optimizer=tf.compat.v2.optimizers.Adam(), loss=tf.compat.v2.losses.mean_squared_error):
        self.batch_size = batch_size
        self.model = dqsa(usernet, input_size, batch_size=batch_size)
        self.model.compile(optimizer=optimizer, loss=loss)
        self.optimizer = self.model.optimizer

    def get_batch_size(self):
        return self.batch_size

    @tf.function
    def __call__(self, inputs):
        return self.model(inputs)

    def fit(self, lr, ER: ExperienceReplay, centralTarget):
        """
        fitting the model, the current version evaluates the target vector for every time step (while taking into
        consideration the sequence leading to that time step), the target vector is evaluated using the latest policy
        once we evaluate the entire target vector, we apply backprop with that target vector
        :param lr: learning rate
        :param ER: Experience Replay
        :param centralTarget: central target
        :return: mean loss value
        """
        lossValue = []
        self.model.optimizer.learning_rate = lr  # deciding the optimizer learning rate
        for _ in range(config.train_iterations):
            seq_len = config.TimeSlots  # following the paper
            #  seq_len = randrange(start=5, stop=config.TimeSlots)
            exp_batch = ER.getMiniBatch(batch_size=config.batch_size, seq_length=seq_len)
            states_processed, actions_processed, next_states_processed, rewards_processed = processing_experience(exp_batch)
            # done organizing data to shape (number_of_users * batch size , 50, 2K + 2)
            target_vector = self(states_processed)
            target_vector = target_vector.numpy()
            next_state_qvalues = self(next_states_processed)
            next_state_qvalues = next_state_qvalues[:, 1:, :].numpy()
            next_state_qvalues_target = centralTarget(next_states_processed)
            next_state_qvalues_target = next_state_qvalues_target[:, 1:, :].numpy()
            evaluated_actions = np.argmax(next_state_qvalues, axis=-1).astype(np.int32)
            for t in range(seq_len):
                double_dqn = np.asarray([Qvalue[evaluated_actions[i, t]] for i, Qvalue in enumerate(next_state_qvalues_target[:, t, :])])
                for i, nextQvalue in enumerate(double_dqn):  # creating the labels
                    target_vector[i, t, int(actions_processed[i, t])] = rewards_processed[i, t] + config.Gamma * nextQvalue
            loss = self.model.train_on_batch(x=states_processed, y=target_vector)
            lossValue.append(loss)
        return np.mean(lossValue)

    def reset_states(self):
        self.model.reset_states()

    def load_weights(self, path):
        self.model.load_weights(filepath=path)

    def save_weights(self, path):
        self.model.save_weights(filepath=path)

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    # @tf.function
    # def trainPhase(self, target_vector, states):
    #          with tf.GradientTape() as tape:
    #              current_predictions = self(states)
    #              train_loss = self.model.loss(target_vector, current_predictions)
    #          grads = tape.gradient(train_loss, self.model.trainable_variables)
    #          return train_loss, grads
    #
    # @tf.function
    # def applyGradients(self, grads):
    #          self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
