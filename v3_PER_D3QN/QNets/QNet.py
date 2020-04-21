import datetime
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

import keras.backend as k
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, Add, Subtract, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from v3_PER_D3QN.ReplayMemory import Memory as m

DECAY = 0.995
BATCH_SIZE = 32
DISCOUNT = 0.85
LEARNING_RATE = 0.005
SAVE_DIR = 'SavedModels'
MEMORY_CAPACITY = int(50e3)


class QNet:
    def __init__(self, input_shape, output_shape, model_name):
        self.model_name = model_name
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.target_update_counter = 0

        self.gamma = DISCOUNT
        self.save_dir = SAVE_DIR
        self.epsilon_decay = DECAY
        self.batch_size = BATCH_SIZE
        self.learning_rate = LEARNING_RATE
        self.replay_memory = m.Memory(MEMORY_CAPACITY)

        self.model_path = self.save_dir + '/' + self.model_name

        self.tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir="logs/{}-{}".format(self.model_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

        self.model, self.target_model = self.load_mod()
        self.save_model()
        self.train_target_model()

    def load_mod(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        if os.path.exists(self.model_path):
            return load_model(self.model_path), \
                   self.create_model()
        else:
            print('No Model Found | Creating New D3QN Model')
            return self.create_model(), self.create_model()

    def save_model(self):
        self.model.save(self.model_path, save_format='tf')

    def train_target_model(self):
        # TODO: check tau training
        if self.target_model is not None:
            self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state, action_verbose):
        # TODO: New EG Strategy
        if state.ndim is 3:
            state = np.expand_dims(state, axis=0)
        self.epsilon *= self.epsilon_decay
        # noinspection PyAttributeOutsideInit
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            action = random.randint(0, self.output_shape - 1)
            return action
        action = np.argmax(self.model.predict(state)[0])
        if action_verbose is 1:
            print('Predicted Action:', action + 1, 'Epsilon:', self.epsilon, self.model.predict(state))
        return action

    def update_replay_memory(self, transition):
        self.replay_memory.store(transition)

    def create_model(self):
        # TODO: Add LSTM and maybe dropout
        inp = Input(self.input_shape)
        out = Conv2D(data_format='channels_last',
                     filters=256,
                     kernel_size=(3, 3),
                     strides=(2, 2),
                     activation='relu')(inp)
        out = MaxPooling2D()(out)
        out = Conv2D(data_format='channels_last',
                     filters=256,
                     kernel_size=(3, 3),
                     strides=(2, 2),
                     activation='relu')(out)
        out = MaxPooling2D()(out)
        out = Flatten()(out)
        value = Dense(units=64,
                      kernel_initializer='zeros',
                      activation='relu')(out)
        value = Dense(units=1,
                      kernel_initializer='zeros',
                      activation='relu')(value)
        advantage = Dense(units=64,
                          kernel_initializer='zeros',
                          activation='relu')(out)
        advantage = Dense(units=self.output_shape,
                          kernel_initializer='zeros',
                          activation='softmax')(advantage)
        advantage_mean = Lambda(lambda x: k.mean(x, axis=1))(advantage)
        advantage = Subtract()([advantage, advantage_mean])
        out = Add()([value, advantage])

        model = Model(inputs=inp, outputs=out)
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def train_model(self, terminal_state):
        tree_index, batch, is_weights = self.replay_memory.sample(self.batch_size)

        states_mb = np.array([each[0][0] for each in batch])
        actions_mb = np.array([each[0][1] for each in batch])
        rewards_mb = np.array([each[0][2] for each in batch])
        next_states_mb = np.array([each[0][3] for each in batch])
        dones_mb = np.array([each[0][4] for each in batch])

        q_cs = self.model.predict(states_mb)
        q_cs_old = np.array(q_cs)
        q_ns = self.model.predict(next_states_mb)
        q_t_ns = self.target_model.predict(next_states_mb)

        x = [states_mb]
        y = []
        abs_errors = []

        for i in range(0, len(batch)):
            terminal = dones_mb[i]

            action = np.argmax(self.model.predict(q_ns[i])[0])

            if not terminal:
                target = rewards_mb[i] + self.gamma * q_t_ns[i][action]
            else:
                target = rewards_mb[i]

            current_q = q_cs[i]
            current_q[action] = target
            y.append(current_q)
            abs_errors.append(q_cs_old[i, action] - current_q[i, action])

        self.replay_memory.batch_update(tree_index, np.abs(abs_errors))

        self.model.fit(np.array(x), np.array(y), batch_size=self.batch_size, verbose=0, shuffle=False,
                       callbacks=[self.tensorboard] if terminal_state else None)
        self.save_model()
