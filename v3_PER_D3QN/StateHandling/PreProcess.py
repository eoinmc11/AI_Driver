import cv2
import numpy as np


# ========== Pre Processing Functions ==========


def conv_2_gray_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def conv_2_hsv(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    return img_hsv, h, s, v


def crop_bottom(img):
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
