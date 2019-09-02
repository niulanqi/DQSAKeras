import tensorflow as tf
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.activations import *
from config import config
from Memory import ExperienceReplay
import numpy as np


# def dqsa(input_size, stateful):
#     inputs = Input(batch_shape=input_size)
#     lstm = LSTM(units=config.LstmUnits, stateful=stateful, return_sequences=False)
#     lstmOutput = lstm(inputs)
#     streamAC = Dense(units=10)(lstmOutput)
#     streamVC = Dense(units=10)(lstmOutput)
#     advantage = Dense(units=config.Actions)(streamAC)
#     value = Dense(units=1)(streamVC)
#     output = value * tf.ones_like(advantage) + tf.subtract(advantage, tf.reduce_mean(advantage, keepdims=True, axis=-1))
#     model = Model(inputs=inputs, outputs=output)
#     return model


class DQSA(tf.keras.Model):
    def __init__(self, input_size, usernet):
        """
        creating an instance of a DQSA model
        :param input_size: input size of the network
        :param usernet: a signal to determine if the network should be stateful or not
               (usernet = stateful)
        """
        super(DQSA, self).__init__()
        self.lstm = LSTM(units=config.LstmUnits, stateful=usernet, return_sequences=False, batch_input_shape=input_size)
        self.streamAC = Dense(units=10, activation=relu)
        self.streamVC = Dense(units=10, activation=relu)
        self.advantage = Dense(units=config.Actions)
        self.value = Dense(units=1)
        # according to the matlab script we need to get rid of the biases for the deterministic values


    @tf.function
    def call(self, inputs):
        """
        forward pass
        :param inputs: input states to the DQSA
        :return: prediction of shape K + 1
        """
        # pred = self.model(inputs)
        lstmOutput = self.lstm(inputs)
        streamVC = self.streamVC(lstmOutput)
        streamAC = self.streamAC(lstmOutput)
        advantage = self.advantage(streamAC)
        value = self.value(streamVC)
        pred = value * tf.ones_like(advantage) + tf.subtract(advantage,
                                                             tf.reduce_mean(advantage, keepdims=True, axis=-1))
        return pred

    def define_optimizer(self, optimizer):
        self.optimizer = optimizer

    def define_loss(self, loss):
        self.loss = loss


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
        self.optimizer.learning_rate = lr  # deciding the optimizer learning rate
        for _ in range(config.train_iterations):  # usually one
            seq_len = 50  # following the paper
            exp_batch = ER.getMiniBatch(batch_size=config.batch_size, seq_length=seq_len)
            states = np.squeeze(np.asarray([exp.state for exp in exp_batch]))
            actions = np.squeeze(np.asarray([exp.action for exp in exp_batch]))
            next_states = np.squeeze(np.asarray([exp.next_state for exp in exp_batch]))
            rewards = np.squeeze(np.asarray([exp.reward for exp in exp_batch]))
            # next_states = np.concatenate((np.expand_dims(states[:, :, 0, :], axis=2), next_states), axis=2)
            # concatenating the first state to the next state sequence to get the "NEXT STATE" expression
            # reshaping the experiences to be ( number_of_users * batch size , 50/51, 2K + 2)
            # the current version does not support this because we currently use a fixed initial state and not a
            # randomized one
            states_processed = np.reshape(states, newshape=[-1, states.shape[2], states.shape[3]])
            actions_processed = np.reshape(actions, [-1, actions.shape[2]])
            rewards_processed = np.reshape(rewards, [-1, rewards.shape[2]])
            next_states_processed = np.reshape(next_states, newshape=[-1, next_states.shape[2], next_states.shape[3]])
            # done organizing data to shape (number_of_users * batch size , 50, 2K + 2)
            grads = []
            for t in range(1, seq_len):   # going through the time steps we skip the initial state which is fixed
                # according to the matlab code, we got time step by time step and extract the target vector
                target_next_state_qvalues = centralTarget(next_states_processed[:, : t + 1, :])
                # no initial fixed state so we dont need to skip the zero element
                next_state_qvalues = self(next_states_processed[:, :t + 1, :])
                evaluated_actions = np.argmax(next_state_qvalues, axis=-1).astype(np.int32)
                double_dqn = np.asarray([Qvalue[evaluated_actions[i]] for i, Qvalue in enumerate(target_next_state_qvalues)])
                target_vector = rewards_processed[:, t] + config.Gamma * double_dqn  # bellman equation with double dqn
                one_hot_actions = tf.compat.v1.one_hot(actions_processed[:, t].astype(np.int32), depth=config.Actions)
                with tf.GradientTape() as tape:  # custom training
                    current_predictions = self(states_processed[:, 1: t + 1, :])
                    current_predictions_at_action_chosen = tf.reduce_sum(tf.multiply(current_predictions,
                                                                                     one_hot_actions), axis=1)
                    train_loss = self.loss(target_vector, current_predictions_at_action_chosen)
                # evaluating the gradients with the latest policy
                grads.append(tape.gradient(train_loss, self.trainable_variables))
                lossValue.append(train_loss)
            # applying the gradients
            [self.optimizer.apply_gradients(zip(gradsEle, self.trainable_variables)) for gradsEle in grads]
        return np.mean(lossValue)

