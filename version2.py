# Import Libraries
import os
import cv2
import sys
import math
import random
import datetime
import numpy as np
import tensorflow as tf
import keras.backend as k
import matplotlib.pyplot as plt

import pyglet
from pyglet import gl
from pyglet.window import key

import gym
from gym import spaces
from gym.wrappers.monitor import Monitor
from gym.envs.box2d.car_dynamics import Car
from gym.utils import colorize, seeding, EzPickle

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input, Add, Subtract, Lambda

# Import Project Files
from Enviroments import CartPole

# Reduce logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ========== Game Actions ==========

# car_actions = (np.array([-1.0, 0.0, 0.0]),  # 1. Full Left
#                np.array([+1.0, 0.0, 0.0]),  # 2. Full Right
#                np.array([-0.5, +0.5, 0.0]),  # 3. Half Left, Half Acceleration
#                np.array([+0.5, +0.5, 0.0]),  # 4. Half Right, Half Acceleration
#                np.array([0.0, +1.0, 0.0]),  # 5. Full Acceleration
#                np.array([0.0, +0.5, 0.0]),  # 6. Half Acceleration
#                np.array([0.0, 0.0, 0.6]),  # 7. 60% Brake
#                np.array([0.0, 0.0, 0.3]),  # 8. 30% Brake
#                )

# ========== Game Params ==========

batch_size = 32
episode_num = 1
training = True
action_verbose = 1
record_video = False
human_control = True
train_target_every = 5
model_name = "PER_TEST"

min_reward_allowed = -20.0  # Finish Episode if this is reached

# ========== Racer Params ==========

STATE_W = 96
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 2.7  # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)

TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4

ROAD_COLOR = [0.4, 0.4, 0.4]

off_track_deduction = -25


# ========== Play The Game ==========

def record_game(env):
    # TODO: Test this
    return Monitor(env, '/tmp/video-test', force=True)


def ai_driver():
    env = CarRacing()
    env.render()
    current_state = env.reset()
    input_shape = current_state.shape  # Neural Network input shape
    output_shape = len(env.car_actions)
    agent = DQN(input_shape, output_shape, batch_size, model_name)
    if record_video:
        env = record_game(env)
    game_is_running = True

    while game_is_running:
        game_current_state = env.reset()
        total_reward = 0.0
        restart, done, reward, steps, episode_num = False, False, 0, 0, 1
        new_state = game_current_state
        while True:
            # Get Action to take based on the Current State
            act = agent.get_action(game_current_state, action_verbose)
            action = env.car_actions[act]

            if steps > 40:

                new_state, reward, done, info = env.step(action)
                agent.per_memory.store([game_current_state, act, reward, new_state, done])
            if training:
                agent.train_model_per(done)

            # Update params and check game status
            total_reward += reward
            steps += 1
            game_is_running = env.render()
            game_current_state = new_state

            if done or not game_is_running or restart or total_reward < min_reward_allowed:
                agent.save_model()
                episode_num += 1
                if episode_num % train_target_every == 0:
                    agent.train_target_model()
                break


def human_driver():
    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == 0xff0d: restart = True
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

    env = CarRacing()
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    if record_video:
        env = record_game(env)
    game_is_running = True

    while game_is_running:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(a)
            total_reward += r
            steps += 1
            game_is_running = env.render()
            if done or restart or game_is_running is False:
                break
    env.close()


# ========== Pre Processing Functions ==========

def conv_2_gray_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def conv_2_hsv(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    return img_hsv, h, s, v


def crop_bottom(img):
    plt.imshow(img[:81, :])
    plt.show()
    return img[:81, :]


def on_track_detection(img):
    """Image needs to be a single channel"""
    detection = []
    off_track_h_val = 60

    left_pt = img[72, 43]
    right_pt = img[72, 52]
    front_l_pt = img[65, 46]
    front_r_pt = img[65, 50]

    detection.append(1 if int(left_pt) is off_track_h_val else 0)
    detection.append(1 if int(right_pt) is off_track_h_val else 0)
    detection.append(1 if int(front_r_pt) is off_track_h_val else 0)
    detection.append(1 if int(front_l_pt) is off_track_h_val else 0)
    return False if sum(detection) >= 3 else True


def state_process(img):
    _, state_h, _, _ = conv_2_hsv(crop_bottom(img))
    return np.expand_dims(state_h, axis=2)  # Increase dim to 4 as prediction needs epoch dim


# ========== Racer ==========

class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        tile.color[0] = ROAD_COLOR[0]
        tile.color[1] = ROAD_COLOR[1]
        tile.color[2] = ROAD_COLOR[2]
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            # print tile.road_friction, "ADD", len(obj.tiles)
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += 1000.0 / len(self.env.track)
                self.env.tile_visited_count += 1
        else:
            obj.tiles.remove(tile)
            # print tile.road_friction, "DEL", len(obj.tiles) -- should delete to zero when on grass (this works)


class CarRacing(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second': FPS
    }

    def __init__(self, verbose=1):
        EzPickle.__init__(self)
        self.seed()
        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose
        self.track_steps = 0
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=
                               [(0, 0), (1, 0), (1, -1), (0, -1)]))

        # TODO: redo
        self.action_space = spaces.Box(np.array([-1, 0, 0]), np.array([+1, +1, +1]),
                                       dtype=np.float32)  # steer, gas, brake
        self.car_actions = (np.array([-1.0, 0.0, 0.0]),  # 1. Full Left
                            np.array([+1.0, 0.0, 0.0]),  # 2. Full Right
                            np.array([-0.5, +0.5, 0.0]),  # 3. Half Left, Half Acceleration
                            np.array([+0.5, +0.5, 0.0]),  # 4. Half Right, Half Acceleration
                            np.array([0.0, +1.0, 0.0]),  # 5. Full Acceleration
                            np.array([0.0, +0.5, 0.0]),  # 6. Half Acceleration
                            np.array([0.0, 0.0, 0.6]),  # 7. 60% Brake
                            np.array([0.0, 0.0, 0.3]),  # 8. 30% Brake
                            )
        self.observation_space = spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        self.car.destroy()

    def _create_track(self):
        CHECKPOINTS = 12

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            alpha = 2 * math.pi * c / CHECKPOINTS + self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)
            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * TRACK_RAD
            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
        # print "\n".join(str(h) for h in checkpoints)
        # self.road_poly = [ (    # uncomment this to see checkpoints
        #    [ (tx,ty) for a,tx,ty in checkpoints ],
        #    (0.7,0.7,0.9) ) ]
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi
            while True:  # Find destination from checkpoints
                failed = True
                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break
                if not failed:
                    break
                alpha -= 2 * math.pi
                continue
            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            proj = r1x * dest_dx + r1y * dest_dy  # destination vector projected on rad
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break
        # print "\n".join([str(t) for t in enumerate(track)])

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose == 1:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))

        self.tilez = i2 - i1
        assert i1 != -1
        assert i2 != -1

        track = track[i1:i2 - 1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2])) +
            np.square(first_perp_y * (track[0][3] - track[-1][3])))
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (x1 - TRACK_WIDTH * math.cos(beta1), y1 - TRACK_WIDTH * math.sin(beta1))
            road1_r = (x1 + TRACK_WIDTH * math.cos(beta1), y1 + TRACK_WIDTH * math.sin(beta1))
            road2_l = (x2 - TRACK_WIDTH * math.cos(beta2), y2 - TRACK_WIDTH * math.sin(beta2))
            road2_r = (x2 + TRACK_WIDTH * math.cos(beta2), y2 + TRACK_WIDTH * math.sin(beta2))
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3)
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            t.road_visited = False
            t.road_friction = 1.0
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (x1 + side * TRACK_WIDTH * math.cos(beta1), y1 + side * TRACK_WIDTH * math.sin(beta1))
                b1_r = (x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                        y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1))
                b2_l = (x2 + side * TRACK_WIDTH * math.cos(beta2), y2 + side * TRACK_WIDTH * math.sin(beta2))
                b2_r = (x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                        y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2))
                self.road_poly.append(([b1_l, b1_r, b2_r, b2_l], (1, 1, 1) if i % 2 == 0 else (1, 0, 0)))
        self.track = track
        return True

    def get_tiles(self):
        print(self.tilez)
        return self.tilez

    def reset(self):
        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.road_poly = []
        self.track_steps = 0

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose == 1:
                print("retry to generate track (normal if there are not many of this messages)")
        self.car = Car(self.world, *self.track[0][1:4])

        return self.step(None)[0]

    def step(self, action):  # Added extra parameter to check if car has left the track
        if action is not None:
            self.car.steer(-action[0])
            self.car.gas(action[1])
            self.car.brake(action[2])

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        self.state = state_process(self.render("state_pixels"))

        step_reward = 0
        done = False
        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward

            # Check if car is on track
            if not on_track_detection(self.state) and self.track_steps > 40:
                done = True
                step_reward = off_track_deduction

            if self.tile_visited_count == len(self.track):
                done = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                done = True
                step_reward = -100

        self.track_steps += 1
        return self.state, step_reward, done, {}

    def render(self, mode='human'):
        assert mode in ['human', 'state_pixels', 'rgb_array']
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label('0000', font_size=36,
                                                 x=20, y=WINDOW_H * 2.5 / 40.00, anchor_x='left', anchor_y='center',
                                                 color=(255, 255, 255, 255))
            self.transform = rendering.Transform()

        if "t" not in self.__dict__:
            return

        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)  # Animate zoom first second
        zoom_state = ZOOM * SCALE * STATE_W / WINDOW_W
        zoom_video = ZOOM * SCALE * VIDEO_W / WINDOW_W
        scroll_x = self.car.hull.position[0]
        scroll_y = self.car.hull.position[1]
        angle = -self.car.hull.angle
        vel = self.car.hull.linearVelocity
        if np.linalg.norm(vel) > 0.5:
            angle = math.atan2(vel[0], vel[1])
        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(
            WINDOW_W / 2 - (scroll_x * zoom * math.cos(angle) - scroll_y * zoom * math.sin(angle)),
            WINDOW_H / 4 - (scroll_x * zoom * math.sin(angle) + scroll_y * zoom * math.cos(angle)))
        self.transform.set_rotation(angle)

        self.car.draw(self.viewer, mode != "state_pixels")

        arr = None
        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()

        win.clear()
        t = self.transform
        if mode == 'rgb_array':
            VP_W = VIDEO_W
            VP_H = VIDEO_H
        elif mode == 'state_pixels':
            VP_W = STATE_W
            VP_H = STATE_H
        else:
            pixel_scale = 1
            if hasattr(win.context, '_nscontext'):
                pixel_scale = win.context._nscontext.view().backingScaleFactor()  # pylint: disable=protected-access
            VP_W = int(pixel_scale * WINDOW_W)
            VP_H = int(pixel_scale * WINDOW_H)

        gl.glViewport(0, 0, VP_W, VP_H)
        t.enable()
        self.render_road()
        for geom in self.viewer.onetime_geoms:
            geom.render()
        self.viewer.onetime_geoms = []
        t.disable()
        self.render_indicators(WINDOW_W, WINDOW_H)

        if mode == 'human':
            win.flip()
            return self.viewer.isopen

        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep='')
        arr = arr.reshape(VP_H, VP_W, 4)
        arr = arr[::-1, :, 0:3]

        return arr

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def render_road(self):
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(0.4, 0.8, 0.4, 1.0)
        gl.glVertex3f(-PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, -PLAYFIELD, 0)
        gl.glVertex3f(-PLAYFIELD, -PLAYFIELD, 0)
        gl.glColor4f(0.4, 0.9, 0.4, 1.0)
        k = PLAYFIELD / 20.0
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                gl.glVertex3f(k * x + k, k * y + 0, 0)
                gl.glVertex3f(k * x + 0, k * y + 0, 0)
                gl.glVertex3f(k * x + 0, k * y + k, 0)
                gl.glVertex3f(k * x + k, k * y + k, 0)
        for poly, color in self.road_poly:
            gl.glColor4f(color[0], color[1], color[2], 1)
            for p in poly:
                gl.glVertex3f(p[0], p[1], 0)
        gl.glEnd()

    def render_indicators(self, W, H):
        gl.glBegin(gl.GL_QUADS)
        s = W / 40.0
        h = H / 40.0
        gl.glColor4f(0, 0, 0, 1)
        gl.glVertex3f(W, 0, 0)
        gl.glVertex3f(W, 5 * h, 0)
        gl.glVertex3f(0, 5 * h, 0)
        gl.glVertex3f(0, 0, 0)

        def vertical_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place + 0) * s, h + h * val, 0)
            gl.glVertex3f((place + 1) * s, h + h * val, 0)
            gl.glVertex3f((place + 1) * s, h, 0)
            gl.glVertex3f((place + 0) * s, h, 0)

        def horiz_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place + 0) * s, 4 * h, 0)
            gl.glVertex3f((place + val) * s, 4 * h, 0)
            gl.glVertex3f((place + val) * s, 2 * h, 0)
            gl.glVertex3f((place + 0) * s, 2 * h, 0)

        true_speed = np.sqrt(np.square(self.car.hull.linearVelocity[0]) + np.square(self.car.hull.linearVelocity[1]))
        vertical_ind(5, 0.02 * true_speed, (1, 1, 1))
        vertical_ind(7, 0.01 * self.car.wheels[0].omega, (0.0, 0, 1))  # ABS sensors
        vertical_ind(8, 0.01 * self.car.wheels[1].omega, (0.0, 0, 1))
        vertical_ind(9, 0.01 * self.car.wheels[2].omega, (0.2, 0, 1))
        vertical_ind(10, 0.01 * self.car.wheels[3].omega, (0.2, 0, 1))
        horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle, (0, 1, 0))
        horiz_ind(30, -0.8 * self.car.hull.angularVelocity, (1, 0, 0))
        gl.glEnd()
        self.score_label.text = "%04i" % self.reward
        self.score_label.draw()


# ========== DQN ==========

class DQN:
    def __init__(self, input_shape, output_shape, batch_size, model_name):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.model_name = model_name

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.85
        self.learning_rate = 0.005
        self.target_update_counter = 0

        self.save_dir = '../AI_Driver_v0.2/SavedModels'
        self.model_path = self.save_dir + '/' + self.model_name

        self.per_memory = Memory(int(2e4))
        self.tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir="logs/{}-{}".format(self.model_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

        self.model = self.load_model()
        self.target_model = self.create_model()
        self.save_model()
        self.train_target_model()

    def load_model(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
            print("Directory to store models created")

        if os.path.exists(self.model_path):
            return load_model(self.model_path)
        else:
            return self.create_model()

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
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model

    def save_model(self):
        self.model.save(self.model_path, save_format='tf')

    def train_target_model(self):
        # TODO: check tau training
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
        self.per_memory.store(transition)

    def ttr(self, terminal_state):

        tree_index, minibatch = self.per_memory.sample(self.batch_size)

        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch]) / 255
        future_qs_list = self.target_model.predict(new_current_states)

        actions = np.array([transition[1] for transition in minibatch])
        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.gamma * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        indices = np.arange(self.batch_size, dtype=np.int32)
        absolute_errors = np.abs(current_qs_list[indices, actions] - target[indices, actions])
        # Update priority
        self.per_memory.batch_update(tree_index, absolute_errors)


        self.model.fit(np.array(X) / 255, np.array(y), batch_size=self.batch_size, verbose=0, shuffle=False,
                       callbacks=[self.tensorboard] if terminal_state else None)
        self.save_model()

    def train_model_per(self, terminal_state):
        # TODO: FIX
        tree_index, minibatch = self.per_memory.sample(self.batch_size)

        current_state = np.zeros((self.batch_size, self.input_shape))
        next_state = np.zeros((self.batch_size, self.input_shape))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            current_state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        # predict Q-values for starting state using the main network
        current_qs = self.model.predict(current_state)
        target_old = np.array(current_qs)
        # predict best action in ending state using the main network
        target_next = self.model.predict(next_state)
        # predict Q-values for ending state using the target network
        target_qs = self.target_model.predict(next_state)

        for i in range(len(minibatch)):
            # correction on the Q value for the action used
            if done[i]:
                current_qs[i][action[i]] = reward[i]
            else:
                # current Q Network selects the action
                # a'_max = argmax_a' Q(s', a')
                a = np.argmax(target_next[i])
                # target Q Network evaluates the action
                # Q_max = Q_target(s', a'_max)
                current_qs[i][action[i]] = reward[i] + self.gamma * (target_qs[i][a])

        indices = np.arange(self.batch_size, dtype=np.int32)
        absolute_errors = np.abs(target_old[indices, np.array(action)] - current_qs[indices, np.array(action)])
        # Update priority
        self.per_memory.batch_update(tree_index, absolute_errors)

        self.model.fit(current_state, current_qs, batch_size=self.batch_size, verbose=0, shuffle=False,
                       callbacks=[self.tensorboard] if terminal_state else None)
        self.save_model()


class SumTree(object):
    data_pointer = 0  # Which leaf are we are (L to R) | Index in data

    def __init__(self, capacity):
        self.capacity = capacity  # Trees capacity
        self.tree = np.zeros(2 * capacity - 1)  # Tree
        self.data = np.zeros(capacity, dtype=object)  # for all transitions

    def add_experience(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1  # What index to put priority
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_index, priority)  # update tree_frame

        # Go back to the start if full
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]  # Change = new priority score - former priority score
        self.tree[tree_index] = priority  # Set new priority
        # then propagate the change through tree
        while tree_index != 0:  # this method is faster than the recursive loop
            tree_index = (tree_index - 1) // 2  # Accesses the parent leaf
            self.tree[tree_index] += change

    def get_leaf(self, v):
        """
        SEARCH FOR PRIORITY AND EXPERIENCE

        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
           1   2
          / \ / \
         3  4 5  6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_index = 0
        while True:  # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1  # this leaf's left and right kids
            right_child_index = left_child_index + 1
            if left_child_index >= len(self.tree):  # reach bottom, end search
                leaf_index = parent_index
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    per_epsilon = 0.01  # small amount to avoid zero priority
    per_alpha = 0.6  # [0~1] convert the importance of TD error to priority
    per_beta = 0.4  # importance-sampling, from initial value increasing to 1
    per_beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        # Find Max Priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = self.abs_err_upper
        self.tree.add_experience(max_priority, transition)  # set the max p for new p

    def sample(self, n):
        minibatch = []
        batch_index = np.empty((n,), dtype=np.int32)

        # Calculate the priority segment
        priority_segment = self.tree.total_priority / n  # priority segment

        for i in range(n):
            # A value is uniformly sample from each range
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            # Experience that correspond to each value is retrieved
            index, priority, data = self.tree.get_leaf(value)

            batch_index[i] = index

            minibatch.append([data[0], data[1], data[2], data[3], data[4]])

        return batch_index, minibatch

    def batch_update(self, tree_index, abs_errors):
        abs_errors += self.per_epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.per_alpha)
        for ti, p in zip(tree_index, ps):
            self.tree.update(ti, p)


if __name__ == '__main__':

    if human_control:
        print("Human is Driving")
        human_driver()
    elif not human_control:
        print("AI is Driving")
        ai_driver()
