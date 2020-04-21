import datetime
import os
import random
from _collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from v2.QNets import DuellingDQN, DQN
from v2.ReplayMemory import Memory as m

DECAY = 0.995
BATCH_SIZE = 32
DISCOUNT = 0.85
SAVE_DIR = 'SavedModels'
MEMORY_CAPACITY = int(50e3)


class QNet:
    def __init__(self, input_shape, output_shape, model_name, per, duelling, double):
        self.per = per
        self.double = double
        self.duelling = duelling
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
        self.replay_memory = m.Memory(MEMORY_CAPACITY) if self.per else deque(maxlen=MEMORY_CAPACITY)
        # self.per_replay_memory = m.Memory(MEMORY_CAPACITY)

        self.model_path = self.save_dir + '/' + self.model_name

        self.tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir="logs/{}-{}".format(self.model_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

        self.model, self.target_model = self.load_mod()
        self.save_model()
        self.train_target_model()

    def load_mod(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        # D3QN
        if self.duelling and self.double:
            if os.path.exists(self.model_path):
                return load_model(self.model_path), \
                       DuellingDQN.DuellingDQN(self.input_shape, self.output_shape).create_model()
            else:
                print('No Model Found | Creating New D3QN Model')
                return DuellingDQN.DuellingDQN(self.input_shape, self.output_shape).create_model(), \
                       DuellingDQN.DuellingDQN(self.input_shape, self.output_shape).create_model()

        # Double DQN
        elif not self.duelling and self.double:
            if os.path.exists(self.model_path):
                return load_model(self.model_path), \
                       DQN.DQN(self.input_shape, self.output_shape).create_model()
            else:
                print('No Model Found | Creating New DoubleDQN Model')
                return DQN.DQN(self.input_shape, self.output_shape).create_model(), \
                       DQN.DQN(self.input_shape, self.output_shape).create_model()

        # Duelling DQN
        elif self.duelling and not self.double:
            if os.path.exists(self.model_path):
                return load_model(self.model_path), None
            else:
                print('No Model Found | Creating New DuellingDQN Model')
                return DuellingDQN.DuellingDQN(self.input_shape, self.output_shape).create_model(), None

        # DQN
        elif not self.duelling and not self.double:
            if os.path.exists(self.model_path):
                return load_model(self.model_path), None
            else:
                print('No Model Found | Creating New DQN Model')
                return DQN.DQN(self.input_shape, self.output_shape).create_model(), None

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
        if self.per:
            self.replay_memory.store(transition)
        else:
            self.replay_memory.append(transition)

    def train_model(self, terminal_state):
        # Check if PER or ER is being used
        global tree_index
        if self.per:
            tree_index, minibatch = self.replay_memory.sample(self.batch_size)
        else:
            if len(self.replay_memory) < self.batch_size:
                return
            minibatch = random.sample(self.replay_memory, self.batch_size)
            print(np.array(minibatch).shape)
            exit()

        # Gets Q-Values from the Current States
        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_qs_list = self.model.predict(current_states)
        target_old = np.array(current_qs_list)

        # Get the Q-Values from the Next States
        if self.target_model is not None:  # If Double DQN is being used
            new_states = np.array([transition[3] for transition in minibatch]) / 255
            future_qs_list = self.target_model.predict(new_states)

        else:  # If Double DQN is not being used
            new_states = np.array([transition[3] for transition in minibatch]) / 255
            future_qs_list = self.model.predict(new_states)

        x = []
        y = []
        abs_errors = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.gamma * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            x.append(current_state)
            y.append(current_qs)
            if self.per:
                abs_errors.append(target_old[index, action] - current_qs[index, action])

        if self.per:
            self.replay_memory.batch_update(tree_index, np.abs(abs_errors))

        self.model.fit(np.array(x) / 255, np.array(y), batch_size=self.batch_size, verbose=0, shuffle=False,
                       callbacks=[self.tensorboard] if terminal_state else None)
        self.save_model()
