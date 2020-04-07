import os
import gym
import random
import datetime
import time
import numpy as np
import tensorflow as tf

from gym import wrappers
from _collections import deque
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input, InputLayer

from TensorBoard import ModifiedTensorBoard as mtb


class DQN:
    def __init__(self, dqn_env, num_actions, batch_size):
        self.env = dqn_env
        self.output_dim = num_actions
        self.replay_memory = deque(maxlen=2000)

        self.gamma = 0.85
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.agent_name = 'DQN'
        self.learning_rate = 0.005
        self.model_directory = 'Models'
        self.target_update_counter = 0
        self.tensorboard = mtb.ModifiedTensorBoard(log_dir="logs/{}-{}".format(self.agent_name, int(time.time())))

        if not os.path.exists(self.model_directory):
            os.mkdir(self.model_directory)

        if os.path.exists(self.model_directory + '/' + self.agent_name):
            print('Model found: Loading.....', '\n')
            self.model = load_model(self.model_directory + '/' + self.agent_name)
        else:
            print('No Model Found: Creating....')
            self.model = self.create_model()
            self.save_model(self.model_directory + '/' + self.agent_name)

        # Create Target Model
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

    def create_model(self):
        model = Sequential()
        state_shape = (96, 96, 1)
        model.add(Conv2D(input_shape=state_shape,
                         data_format='channels_last',
                         filters=256,
                         kernel_size=(3, 3),
                         strides=(2, 2),
                         activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(data_format='channels_last',
                         filters=256,
                         kernel_size=(3, 3),
                         strides=(2, 2),
                         activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(units=64, kernel_initializer='zeros', activation='relu'))
        model.add(Dense(units=self.output_dim, kernel_initializer='zeros', activation='softmax'))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model

    def save_model(self, fn):
        self.model.save(fn, save_format='tf')

    def update_memory(self, transition):
        self.replay_memory.append(transition)

    def target_train(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if state.ndim is 3:
            state = np.expand_dims(state, axis=0)
        self.epsilon *= self.epsilon_decay
        # noinspection PyAttributeOutsideInit
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            action = random.randint(0, self.output_dim - 1)
            # print('Random', action)
            return action
        action = np.argmax(self.model.predict(state)[0])
        print('Predicted', action + 1, self.model.predict(state))
        return action

    def train_model(self, terminal_state):
        if len(self.replay_memory) < self.batch_size:
            return

        samples = random.sample(self.replay_memory, self.batch_size)

        current_states = np.array([transition[0] for transition in samples]) / 255
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in samples]) / 255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(samples):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.gamma * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X) / 255, np.array(y), batch_size=self.batch_size, verbose=0, shuffle=False,
                       callbacks=[self.tensorboard] if terminal_state else None)
        self.model.save(self.model_directory + '/' + self.agent_name)

