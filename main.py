# Import Libraries
import numpy as np
import os
import tensorflow as tf
import datetime

# Import Project Files
from Model import model
from CarRacerEnvs import Racer
from StateHandling import stateProcess as Sp

# Reduce logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":

    # ACTIONS STATE v1
    actions = (np.array([-1.0, 0.0, 0.0]),  # 1. Full Left
               np.array([+1.0, 0.0, 0.0]),  # 2. Full Right
               np.array([-0.5, +0.5, 0.0]),  # 3. Half Left, Half Acceleration
               np.array([+0.5, +0.5, 0.0]),  # 4. Half Right, Half Acceleration
               np.array([0.0, +1.0, 0.0]),  # 5. Full Acceleration
               np.array([0.0, +0.5, 0.0]),  # 6. Half Acceleration
               np.array([0.0, 0.0, 0.6]),   # 7. 60% Brake
               np.array([0.0, 0.0, 0.3]),   # 8. 30% Brake
               )

    num_actions = len(actions)  # Number of possible actions - Neural Net output shape
    batch_size = 16  # Batch size for training the Neural Network and Replay Memory Size
    train_target_every = 5
    train_model_every = 1
    max_steps = 100000

    # Setup/Initialise Environment
    env = Racer.CarRacing()
    env.render()

    _, current_state_h_value, _, _ = Sp.conv_2_hsv(Sp.crop_bottom(env.reset()))
    current_state = np.expand_dims(current_state_h_value, axis=2)  # Increase dim to 4 as prediction needs epoch dim
    input_dim = current_state.shape  # Neural Network input shape

    # Create the Agent
    agent = model.DQN(input_dim, num_actions, batch_size, 2)

    record_video = False
    game_is_running = True

    # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # log_dir = 'logs/dqn/' + current_time
    # summary_writer = tf.summary.create_file_writer(log_dir)

    min_reward_allowed = -20.0  # Finish Episode if this is reached
    episode = 1

    while game_is_running:

        # Reset Env and Params
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        training = True
        on_track = True
        action_verbose = 1
        reward = 0
        action = actions[1]

        print('Episode', episode, 'Starting')

        while True:

            # Get Action to take based on the Current State
            act = agent.get_action(current_state, action_verbose)
            action = actions[act]

            if steps > 40:  # Below ~40 steps the entire screen is not loaded
                on_track = Sp.on_track_detection(current_state)  # Check if agent is on the track

            # Take the action and process new state
            new_state, reward, done, info = env.step(action, on_track)
            _, current_state_h_value, _, _ = Sp.conv_2_hsv(Sp.crop_bottom(new_state))
            new_state = np.expand_dims(current_state_h_value, axis=2)

            agent.update_replay_memory([current_state, act, reward, new_state, done])
            if training and steps % train_model_every is 0:
                agent.train_model(done)

            # Update params and check game status
            total_reward += reward
            steps += 1
            game_is_running = env.render()
            current_state = new_state
            if done or restart or game_is_running is False or total_reward < min_reward_allowed or steps > max_steps:
                agent.save_model()
                episode += 1
                if episode % train_target_every == 0:
                    agent.train_target_model()
                break
    env.close()
