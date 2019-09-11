import tensorflow as tf
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.activations import *
from config import config
from Memory import ExperienceReplay
import numpy as np
from random import randrange


class DQSA(tf.keras.Model):
    def __init__(self, input_size, usernet):
        """
        creating an instance of a DQSA model
        :param input_size: input size of the network
        :param usernet: a signal to determine if the network should be stateful or not
               (usernet = stateful)
        """
        super(DQSA, self).__init__()
        self.lstm = LSTM(units=config.LstmUnits, stateful=usernet, return_sequences=False, batch_input_shape=input_size,
                         input_shape=(None, config.Actions))
        self.streamAC = Dense(units=10, activation=tanh)
        self.streamVC = Dense(units=10, activation=tanh)
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
        grads = []
        seq_len = config.TimeSlots  # following the paper
        #  seq_len = randrange(start=5, stop=config.TimeSlots)
        exp_batch = ER.getMiniBatch(batch_size=config.batch_size, seq_length=seq_len)
        states = np.squeeze(np.asarray([exp.state for exp in exp_batch]))
        actions = np.squeeze(np.asarray([exp.action for exp in exp_batch]))
        next_states = np.squeeze(np.asarray([exp.next_state for exp in exp_batch]))
        rewards = np.squeeze(np.asarray([exp.reward for exp in exp_batch]))
        next_states = np.concatenate((np.expand_dims(states[:, :, 0, :], axis=2), next_states), axis=2)
        # concatenating the first state to the next state sequence to get the "NEXT STATE" expression
        # reshaping the experiences to be ( number_of_users * batch size , 50/51, 2K + 2)
        states_processed = np.reshape(states, newshape=[-1, states.shape[2], states.shape[3]])
        actions_processed = np.reshape(actions, [-1, actions.shape[2]])
        rewards_processed = np.reshape(rewards, [-1, rewards.shape[2]])
        next_states_processed = np.reshape(next_states, newshape=[-1, next_states.shape[2], next_states.shape[3]])
        # done organizing data to shape (number_of_users * batch size , 50, 2K + 2)
        for t in range(seq_len):   # going through the time steps
            # according to the matlab code, we got time step by time step and extract the target vector
            # tstNextState = next_states_processed[:, : t+2, :]
            # tstState = states_processed[:, : t+1, :]
            # tstReward = rewards_processed[:, t]
            # tstActions = actions_processed[:, t]
            target_vector = self(states_processed[:, : t+1, :])
            # until t + 1 with t + 1 not included meaning --> seq from element 0 to element t
            target_next_state_qvalues = centralTarget(next_states_processed[:, : t+2, :])
            next_state_qvalues = self(next_states_processed[:, : t+2, :])
            evaluated_actions = np.argmax(next_state_qvalues, axis=-1).astype(np.int32)
            double_dqn = np.asarray([Qvalue[evaluated_actions[i]] for i, Qvalue in enumerate(target_next_state_qvalues)])
            target_vector = target_vector.numpy()
            # in the matlab code, we take 4 episodes of 50 time steps each (X) and for each time step
            # we calculate the target value(Y), we then sent X and Y ,as labels , to the train function
            # this is exactly what we do here, we evaluate the target vector for each time step and then
            # measure the grads.
            # we apply the gradients ( updating the policy) before we exit the training phase
            for i, nextQvalue in enumerate(double_dqn):  # creating the labels
                target_vector[i, int(actions_processed[i, t])] = rewards_processed[i, t] + config.Gamma * nextQvalue
            train_loss, gradsEle = self.trainPhase(target_vector, states_processed[:, : t+1, :])
            grads.append(gradsEle)
            # if t == 1:
            #     grads = gradsEle
            # else:
            #     for i, grad in enumerate(grads):
            #          grads[i] = self.add(grad, gradsEle[i])

            lossValue.append(train_loss)
        # applying the gradients to change the model's variable --> updating the policy
        # self.applyGradients(grads)
        [self.applyGradients(gradsEle) for gradsEle in grads]
        return np.mean(lossValue)

    @tf.function
    def trainPhase(self, target_vector, states):
        with tf.GradientTape() as tape:
            current_predictions = self(states)
            train_loss = self.loss(target_vector, current_predictions)
        grads = tape.gradient(train_loss, self.trainable_variables)
        return train_loss, grads

    @tf.function
    def applyGradients(self, grads):
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

    @tf.function
    def add(self, a, b):
        return tf.add(a, b)
