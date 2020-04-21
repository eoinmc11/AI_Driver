import os

import numpy as np
from gym.wrappers.monitor import Monitor
from pyglet.window import key

# ========== File Imports ==========
from v2.Enviroments import CarRacingEdited
from v2.QNets import QNet

# ========== Logging ==========
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ========== Boolean ==========
PER = False
DOUBLE = True
DUELING = True
TRAINING = True
AI_DRIVER = True
RECORD_VIDEO = False

# ========== Parameters ==========
TAU = 8
VERBOSE = 1
MIN_REW = -20
MODEL_NAME = 'TEST'


def ai_driver():
    env = CarRacingEdited.CarRacing()
    env.render()
    initial_state = env.reset()
    input_shape = initial_state.shape  # Neural Network input shape
    output_shape = len(env.car_actions)
    agent = QNet.QNet(input_shape, output_shape, MODEL_NAME, PER, DUELING, DOUBLE)
    if RECORD_VIDEO:
        env = record_game(env)

    steps = 0
    episode = 1
    game = True
    while game:
        # Get State | Pre-processing tasks already handled
        current_state = env.reset()
        total_reward = 0.0
        while True:
            act_index = agent.get_action(current_state, VERBOSE)
            action = env.car_actions[act_index]
            new_state, reward, done, info = env.step(action)
            if PER:
                agent.replay_memory.store([current_state, act_index, reward, new_state, done])
            else:
                agent.replay_memory.append([current_state, act_index, reward, new_state, done])
            if TRAINING:
                agent.train_model(done)
            total_reward += reward
            steps += 1
            game = env.render()
            current_state = new_state
            if done or not game or total_reward < MIN_REW:
                agent.save_model()
                episode += 1
                if episode % TAU is 0:
                    agent.train_target_model()
                break


def human_driver():
    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == key.QUESTION:
            restart = True
        if k == key.LEFT:
            a[0] = -1.0
        if k == key.RIGHT:
            a[0] = +1.0
        if k == key.UP:
            a[1] = +1.0
        if k == key.DOWN:
            a[2] = +0.8  # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        if k == key.LEFT and a[0] == -1.0:
            a[0] = 0
        if k == key.RIGHT and a[0] == +1.0:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            a[2] = 0

    env = CarRacingEdited.CarRacing()
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    if RECORD_VIDEO:
        env = record_game(env)

    game = True
    while game:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(a)
            total_reward += r
            steps += 1
            game = env.render()
            if done or restart or game is False:
                break
    env.close()


def record_game(env):
    # TODO: Test this
    return Monitor(env, '/tmp/video-test', force=True)


if __name__ == '__main__':
    if AI_DRIVER:
        print('AI in Control')
        ai_driver()
    else:
        print('Human in Control')
        human_driver()
