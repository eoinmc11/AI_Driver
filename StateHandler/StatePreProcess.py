import cv2
import matplotlib.pyplot as plt


def conv_2_gray_scale(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plt.imshow(img_gray, cmap='gray')
    # plt.show()
    return img_gray


def conv_2_hsv(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    # plt.figure(1)
    # plt.subplot(1, 3, 1)
    # plt.imshow(h, cmap='gray')
    # plt.subplot(1, 3, 2)
    # plt.imshow(s, cmap='gray')
    # plt.subplot(1, 3, 3)
    # plt.imshow(v, cmap='gray')
    # plt.show()
    return img_hsv, h, s, v


def initial_crop(img):
    return img[:80, :]


# img = cv2.imread('img.png')
# img, h, s, v = conv_2_hsv(img)
# plt.imshow(img)
# plt.show()

