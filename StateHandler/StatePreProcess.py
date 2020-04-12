import cv2
import matplotlib.pyplot as plt
import os


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
    # TODO: Crop edges as well possibly
    return img[:80, :]


def state_2_saved(img, step):
    path = 'ML_ON_2'
    if not os.path.exists(path):
        os.mkdir(path)

    img_h = conv_2_gray_scale(img)
    plt.imshow(img_h, cmap='gray')
    print((step - 40)/10)
    plt.savefig(path + '/image' + str(int((step - 40)/10)) + '.png')


def edge_detection(img):
    pass


# if __name__ == '__main__':
#     if not os.path.exists('ML_ON'):
#         os.mkdir('ML')
#     img = cv2.imread('img.png')
#     # img, h, s, v = conv_2_hsv(img)
#     # plt.imshow(img)
#     # plt.show()

