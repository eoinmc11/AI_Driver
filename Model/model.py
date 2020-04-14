import os
import random
import datetime
import numpy as np
import tensorflow as tf
import keras.backend as K

from _collections import deque
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input, Add, Subtract, Lambda


class DQN:
    def __init__(self, in_shape, num_actions, batch_size, model_type):
        self.input_dim = in_shape
        self.output_dim = num_actions
        self.model_type = model_type
        self.agent_name = 'DQ2TEST'
        self.model_directory = 'SavedModels'
        self.model = self.make_model()
        # TODO: Prioritised Experience Replay
        self.replay_memory = deque(maxlen=20000)

        self.gamma = 0.85
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995

        self.learning_rate = 0.005

        self.target_update_counter = 0
        self.tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir="logs/{}-{}".format(self.agent_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        self.target_model.set_weights(self.model.get_weights())

    def make_model(self):
        if self.model_type is 1:
            # Double Deep Q-Net
            if os.path.exists(self.model_directory + '/' + self.agent_name):
                print('Double DQN Model found: Loading.....', '\n')
                self.target_model = self.create_double_dqn_model()
                self.target_model.set_weights(self.model.get_weights())
                mm_model = load_model(self.model_directory + '/' + self.agent_name)
            else:
                print('No Model Found: Creating Double DQN....')
                mm_model = self.create_double_dqn_model()
            self.target_model = self.create_double_dqn_model()
            return mm_model

        if self.model_type is 2:
            # D3QN
            if os.path.exists(self.model_directory + '/' + self.agent_name):
                print('Double DQN Model found: Loading.....', '\n')
                mm_model = load_model(self.model_directory + '/' + self.agent_name)
                self.target_model = self.create_d3qn_model()

            else:
                print('No Model Found: Creating D3QN....')
                mm_model = self.create_d3qn_model()
            self.target_model = self.create_d3qn_model()
            return mm_model

    def create_double_dqn_model(self):
        model = Sequential()
        model.add(Conv2D(input_shape=self.input_dim,
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

    def create_d3qn_model(self):
        # TODO: D3QN
        input = Input(self.input_dim)
        out = Conv2D(data_format='channels_last',
                     filters=256,
                     kernel_size=(3, 3),
                     strides=(2, 2),
                     activation='relu')(input)
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
        advantage = Dense(units=self.output_dim,
                          kernel_initializer='zeros',
                          activation='softmax')(advantage)
        advantage_mean = Lambda(lambda x: K.mean(x, axis=1))(advantage)
        advantage = Subtract()([advantage, advantage_mean])
        out = Add()([value, advantage])

        model = Model(inputs=input, outputs=out)
        model.compile(optimizer="adam", loss="mse")
        return model

    def save_model(self):
        self.model.save(self.model_directory + '/' + self.agent_name, save_format='tf')

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state, action_verbose):
        # TODO: create new epsilon greedy strategy
        if state.ndim is 3:
            state = np.expand_dims(state, axis=0)
        self.epsilon *= self.epsilon_decay
        # noinspection PyAttributeOutsideInit
        self.epsilon = max(self.epsilon_min, self.epsilon)
        # print('Epsilon:', self.epsilon)
        if np.random.random() < self.epsilon:
            action = random.randint(0, self.output_dim - 1)
            # print('Random', action)
            return action
        action = np.argmax(self.model.predict(state)[0])
        if action_verbose:
            print('Predicted Action:', action + 1, 'Epsilon:', self.epsilon, self.model.predict(state))
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
