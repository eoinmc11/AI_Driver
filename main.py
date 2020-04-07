import numpy as np
import os
import gym
import tensorflow as tf
import datetime
from Model import model
from Racer import Racer
from StateHandler import StatePreProcess as spp
from _collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":
    """
    ACTIONS
    1. Full Acceleration
    2. Half Acceleration
    3. 60% Brake
    4. Left with 20% Acceleration
    5. Right with 20% Acceleration
    6. Left with 10% Brake
    7. Right with 10% Brake
    """
    actions = (np.array([0.0, +1.0, 0.0]),
               np.array([0.0, +0.5, 0.0]),
               np.array([0.0, 0.0, +0.6]),
               np.array([-1.0, +0.2, 0.0]),
               np.array([+1.0, +0.2, 0.0]),
               np.array([-1.0, 0.0, +0.1]),
               np.array([+1.0, 0.0, +0.1]),)

    num_actions = len(actions)
    env = Racer.CarRacing()
    env.render()

    current_state = env.reset()
    cropped_state = spp.initial_crop(current_state)
    _, h, _, _ = spp.conv_2_hsv(cropped_state)
    current_state = np.expand_dims(h, axis=2)
    input_dim = current_state.shape
    agent = model.DQN(input_dim, num_actions, 32)

    record_video = False
    isopen = True
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # log_dir = 'logs/dqn/' + current_time
    # summary_writer = tf.summary.create_file_writer(log_dir)

    reward_kill = -20.0
    episode = 1
    while isopen:
        env.reset()
        # tiles = env.get_tiles()
        print('Episode', episode, 'Starting')
        total_reward = 0.0
        steps = 0
        restart = False
        training = True
        reward = 0
        act = 1
        action = actions[act]

        while True:
            if steps % 2 is 0:
                act = agent.get_action(current_state)
                action = actions[act]
            new_state, reward, done, info = env.step(action)
            cropped_state = spp.initial_crop(new_state)
            _, h, _, _ = spp.conv_2_hsv(cropped_state)
            new_state = np.expand_dims(h, axis=2)
            agent.update_memory([current_state, act, reward, new_state, done])
            if training:
                agent.train_model(done)
            total_reward += reward
            steps += 1
            isopen = env.render()
            current_state = new_state
            if done or restart or isopen is False or total_reward < reward_kill:
                if episode % 5 == 0:
                    agent.target_train()
                agent.save_model('Models/DQN')
                episode += 1
                break
    env.close()
