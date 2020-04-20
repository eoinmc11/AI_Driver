import cv2
import matplotlib.pyplot as plt

img = cv2.imread('/Users/eoinmca/PycharmProjects/AI_Driver/hoff.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img[:81, :]

left_pt = (43, 72)
right_pt = (52, 72)
front_l_pt = (46, 65)
front_r_pt = (50, 65)

img = cv2.rectangle(img, left_pt, left_pt, (255, 0, 0), 1)
img = cv2.rectangle(img, right_pt, right_pt, (255, 0, 0), 1)
img = cv2.rectangle(img, front_l_pt, front_l_pt, (255, 0, 0), 1)
img = cv2.rectangle(img, front_r_pt, front_r_pt, (255, 0, 0), 1)

cv2.imwrite('hoffdot.png', img)


plt.imshow(img)
plt.show()


# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# h, s, v = cv2.split(hsv)
# # cv2.imwrite('h.png', h)
#
#
#
# im = cv2.merge((h, h, h, h))
#
# plt.imshow(im, cmap='gray')
# plt.show()