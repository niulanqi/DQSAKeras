import tensorflow as tf
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.activations import *
from config import config
from Memory import ExperienceReplay
import numpy as np


def dqsa(input_size, stateful):
    inputs = Input(batch_shape=input_size)
    lstm = LSTM(units=config.LstmUnits, stateful=stateful, return_sequences=False)
    lstmOutput = lstm(inputs)
    streamAC = Dense(units=10)(lstmOutput)
    streamVC = Dense(units=10)(lstmOutput)
    advantage = Dense(units=config.Actions)(streamAC)
    value = Dense(units=1)(streamVC)
    output = value * tf.ones_like(advantage) + tf.subtract(advantage, tf.reduce_mean(advantage, keepdims=True, axis=-1))
    model = Model(inputs=inputs, outputs=output)
    return model



class DQSA(tf.keras.Model):
    def __init__(self, input_size, usernet):
        super(DQSA, self).__init__()
        # self.stateful = usernet
        self.model = dqsa(input_size, usernet)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        """

        :param inputs: image batches that serve as an input to the net
        :param training: flag for training mode, not relevent for this module
        :return:
        """
        pred = self.model(inputs)
        return pred

    @tf.function
    def compile(self,
              optimizer,
              loss=None,
              metrics=None,
              loss_weights=None,
              sample_weight_mode=None,
              weighted_metrics=None,
              target_tensors=None,
              distribute=None,
              **kwargs):
        self.optimizer = optimizer
        self.loss = loss


    # def fit(self, lr, ER:ExperienceReplay, centralTarget):
    #     """
    #     :param lr: learning rate
    #     :param ER: Experience Replay
    #     :param centralTarget: central target
    #     :return: mean loss value
    #     """
    #     self.optimizer.learning_rate = lr  # deciding the optimizer learning rate
    #     lossValue = []
    #     for _ in range(config.train_iterations):
    #         seq_len = 50
    #         time_window_size = config.usrTimeSteps
    #         start_ptr = 0
    #         end_ptr = time_window_size + start_ptr
    #         exp_batch = ER.getMiniBatch(batch_size=config.batch_size, seq_length=seq_len)
    #         states = np.squeeze(np.asarray([exp.state for exp in exp_batch]))
    #         actions = np.squeeze(np.asarray([exp.action for exp in exp_batch]))
    #         next_states = np.squeeze(np.asarray([exp.next_state for exp in exp_batch]))
    #         rewards = np.squeeze(np.asarray([exp.reward for exp in exp_batch]))
    #         next_states = np.concatenate((np.expand_dims(states[:, :, 0, :], axis=2), next_states), axis=2)
    #         # Done extracting from ER
    #         states_processed = np.reshape(states, newshape=[-1, states.shape[2], states.shape[3]])
    #         actions_processed = np.reshape(actions, [-1, actions.shape[2]])
    #         rewards_processed = np.reshape(rewards, [-1, rewards.shape[2]])
    #         next_states_processed = np.reshape(next_states, [-1, next_states.shape[2], next_states.shape[3]])
    #         # Done organizing the experiences in an efficient way
    #         while end_ptr < seq_len:  # begin training phase with window for time steps
    #             if np.sum(rewards_processed[:, end_ptr]) < config.N:   # dont want sparse experiences
    #                 start_ptr += 1
    #                 end_ptr = time_window_size + start_ptr
    #                 continue
    #             targetNextStateQvalues = centralTarget(next_states_processed[:, start_ptr: end_ptr + 1, :])
    #             NextStateQvalues = self(next_states_processed[:, start_ptr: end_ptr + 1, :])
    #             evaluated_actions = np.argmax(NextStateQvalues, axis=-1).astype(np.int32)
    #             doubleDqn = np.asarray([Qvalue[evaluated_actions[i]] for i, Qvalue in enumerate(targetNextStateQvalues)])
    #             target_vector = rewards_processed[:, end_ptr] + config.Gamma * doubleDqn
    #             #target_vector = rewards_processed[:, end_ptr] + config.Gamma * np.max(targetNextStateQvalues, axis=-1)   # just target DQN
    #             with tf.GradientTape() as tape:
    #                 currentPredictions = self(states_processed)
    #                 tstTmp = actions_processed[:, end_ptr].astype(np.int32)
    #                 one_hot_Actions = tf.compat.v1.one_hot(tstTmp, depth=config.Actions)
    #                 currentPredictionsAtActionChosen = tf.reduce_sum(tf.multiply(currentPredictions, one_hot_Actions), axis=1)
    #                 train_loss = self.loss(target_vector, currentPredictionsAtActionChosen)
    #                 lossValue.append(train_loss)
    #             grads = tape.gradient(train_loss, self.trainable_variables)
    #             self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
    #             start_ptr += 1
    #             end_ptr = time_window_size + start_ptr
    #     return np.mean(lossValue)

    def fit(self, lr, ER: ExperienceReplay, centralTarget):
        """
        :param lr: learning rate
        :param ER: Experience Replay
        :param centralTarget: central target
        :return: mean loss value
        """
        lossValue = []
        self.optimizer.learning_rate = lr  # deciding the optimizer learning rate
        for _ in range(config.train_iterations):  # usually one
            seq_len = 50
            exp_batch = ER.getMiniBatch(batch_size=config.batch_size, seq_length=seq_len)
            states = np.squeeze(np.asarray([exp.state for exp in exp_batch]))
            actions = np.squeeze(np.asarray([exp.action for exp in exp_batch]))
            next_states = np.squeeze(np.asarray([exp.next_state for exp in exp_batch]))
            rewards = np.squeeze(np.asarray([exp.reward for exp in exp_batch]))
            next_states = np.concatenate((np.expand_dims(states[:, :, 0, :], axis=2), next_states), axis=2)
            states_processed = np.reshape(states, newshape=[-1, states.shape[2], states.shape[3]])
            actions_processed = np.reshape(actions, [-1, actions.shape[2]])
            rewards_processed = np.reshape(rewards, [-1, rewards.shape[2]])
            next_states_processed = np.reshape(next_states, [-1, next_states.shape[2], next_states.shape[3]])
            # done organizing data from ER starting to train:
            targetNextStateQvalues = centralTarget(next_states_processed)
            NextStateQvalues = self(next_states_processed)
            evaluated_actions = np.argmax(NextStateQvalues, axis=-1).astype(np.int32)
            doubleDqn = np.asarray([Qvalue[evaluated_actions[i]] for i, Qvalue in enumerate(targetNextStateQvalues)])
            target_vector = rewards_processed[:, -1] + config.Gamma * doubleDqn
            with tf.GradientTape() as tape:
                currentPredictions = self(states_processed)
                tstTmp = actions_processed[:, -1].astype(np.int32)
                one_hot_Actions = tf.compat.v1.one_hot(tstTmp, depth=config.Actions)
                currentPredictionsAtActionChosen = tf.reduce_sum(tf.multiply(currentPredictions, one_hot_Actions),
                                                                 axis=1)
                train_loss = self.loss(target_vector, currentPredictionsAtActionChosen)
                lossValue.append(train_loss)
            grads = tape.gradient(train_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return np.mean(lossValue)




