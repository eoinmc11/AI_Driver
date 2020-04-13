import cv2
import matplotlib.pyplot as plt
import os
import numpy as np


# ========== Used Functions ==========

def conv_2_gray_scale(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray


def conv_2_hsv(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    return img_hsv, h, s, v


def crop_bottom(img):
    # TODO: Crop edges as well possibly
    return img[:80, :]


def on_track_detection(img):
    """Image needs to be a single channel"""
    check_list = []

    left_pt = img[72, 43]
    right_pt = img[72, 52]
    front_l_pt = img[65, 46]
    front_r_pt = img[65, 50]

    check_list.append(1 if int(left_pt) is 60 else 0)
    check_list.append(1 if int(right_pt) is 60 else 0)
    check_list.append(1 if int(front_r_pt) is 60 else 0)
    check_list.append(1 if int(front_l_pt) is 60 else 0)
    on_track = False if sum(check_list) >= 3 else True
    return on_track


# ========== Test Functions ==========


def state_2_saved(img, step):
    path = 'ML_ON_6'
    if not os.path.exists(path):
        os.mkdir(path)

    _, h, _, v = conv_2_hsv(img)
    plt.imshow(h, cmap='gray')
    print((step - 40)/10)
    plt.savefig(path + '/image' + str(int((step - 40)/10)) + '.png')


def edge_detection(img):
    _, img, _, _ = conv_2_hsv(crop_bottom(img))
    img = cv2.rectangle(img, (46, 78), (49, 78), (255, 0, 0), 1)
    img = cv2.rectangle(img, (46, 78), (49, 78), (255, 0, 0), 1)

    img = cv2.rectangle(img, (46, 65), (49, 65), (255, 0, 0), 1)
    img = cv2.rectangle(img, (46, 65), (49, 65), (255, 0, 0), 1)

    img = cv2.rectangle(img, (52, 72), (52, 72), (255, 0, 0), 1)
    img = cv2.rectangle(img, (43, 72), (43, 72), (255, 0, 0), 1)
    plt.imshow(img)
    plt.show()






# img = '/Users/eoinmca/PycharmProjects/AI_Driver/Class_Image_Files/image_2.png'
# img = cv2.imread(img)
# print(on_track_detection(img))



# cwd = os.getcwd()
# lstt = []
# x = 1
# path = '/Users/eoinmca/PycharmProjects/AI_Driver/ML'
# src_path = '/Users/eoinmca/PycharmProjects/AI_Driver/Class_Image_Files'
#
# for dir in os.listdir(path):
#     path2 = path + '/' + dir
#     for file in os.listdir(path2):
#         os.rename(path2 + '/' + file, src_path + '/' + 'image_' + str(x) + '.png')
#         x += 1


