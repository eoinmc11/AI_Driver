import numpy as np
import os
import gym
import tensorflow as tf
import datetime
from Model import model
from Racer import Racer
from _collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":

    reward_memory_length = 20
    reward_memory = deque(maxlen=reward_memory_length)

    def epsilon_greedy(etiles, ereward):
        reward_memory.append(ereward)
        if len(reward_memory) < reward_memory_length:
            print(0.8)
            return 0.8


        min_reward = -0.1 * reward_memory_length
        min_reward -= min_reward
        max_reward = (1000 / etiles) * reward_memory_length + min_reward
        max_reward = int(max_reward)
        reward_sum = sum(reward_memory)
        epsilon_spread = 1 / max_reward
        epsilon = 1 - reward_sum * epsilon_spread
        print(epsilon)
        return epsilon

    actions = (np.array([-1.0, 0.0, 0.0]),
               np.array([-0.5, 0.0, 0.0]),
               np.array([+1.0, 0.0, 0.0]),
               np.array([+0.5, 0.0, 0.0]),
               np.array([0.0, 0.0, 0.0]),
               np.array([0.0, +1.0, 0.0]),
               np.array([0.0, +0.5, 0.0]),
               np.array([0.0, 0.0, +0.8]),
               np.array([0.0, 0.0, +0.3]))
    """
    0. Full Left
    1. Half Left
    2. Full Right
    3. Half Right
    4. Neutral (For steering)
    5. Full Acceleration
    6. Half Acceleration
    7. Full Break (0.8 so its not instant)
    8. Half Break
    """

    num_actions = len(actions)
    env = Racer.CarRacing()
    # env = gym.make('CarRacing-v0')
    env.render()
    current_state = env.reset()
    agent = model.DQN(env, num_actions, 32)
    record_video = False
    isopen = True
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # log_dir = 'logs/dqn/' + current_time
    # summary_writer = tf.summary.create_file_writer(log_dir)

    reward_kill = -20.0
    episode = 1
    while isopen:
        env.reset()
        tiles = env.get_tiles()
        print(tiles)
        print('Episode', episode, 'Starting')
        total_reward = 0.0
        steps = 0
        restart = False
        training = True
        reward = 0

        while True:
            # epsilon = epsilon_greedy(tiles, reward)
            act = agent.get_action(current_state)
            action = actions[act]
            new_state, reward, done, info = env.step(action)
            agent.update_memory([current_state, act, reward, new_state, done])
            if training:
                agent.train_model(done)
            total_reward += reward
            steps += 1
            isopen = env.render()
            current_state = new_state
            if steps % 200 is 0:
                print('Episode Number:', episode, ' | Tot Reward:', total_reward, ' | Steps:', steps)
            if done or restart or isopen is False or total_reward < reward_kill:
                if episode % 5 == 5:
                    agent.target_train()
                agent.save_model('Models/DQN')
                episode += 1
                break
    env.close()
