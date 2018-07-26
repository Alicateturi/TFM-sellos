import cv2
import numpy as np
import copy


def nothing(x):
    pass


cv2.namedWindow('color')

Hmax = 'MaxH'
Hmin = 'MinH'
wnd = 'color'

Smax = 'MaxS'
Smin = 'MinS'
wnd2 = 'color'

Vmax = 'MaxV'
Vmin = 'MinV'
wnd2 = 'color'

# create trackbars for color change
cv2.createTrackbar('Hmin', 'color', 0, 255, nothing)
cv2.createTrackbar('Hmax', 'color', 0, 255, nothing)

cv2.createTrackbar('Smin', 'color', 0, 255, nothing)
cv2.createTrackbar('Smax', 'color', 0, 255, nothing)

cv2.createTrackbar('Vmin', 'color', 0, 255, nothing)
cv2.createTrackbar('Vmax', 'color', 0, 255, nothing)


img = cv2.imread('Sellos/IMG_13.png')

img = cv2.resize(img, (640, 480))

cv2.imshow('original', img)
cv2.waitKey(0)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow('hsv', hsv)
cv2.waitKey(0)

while 1:

    Hmin = cv2.getTrackbarPos('Hmin', 'color')
    Hmax = cv2.getTrackbarPos('Hmax', 'color')

    Smin = cv2.getTrackbarPos('Smin', 'color')
    Smax = cv2.getTrackbarPos('Smax', 'color')

    Vmin = cv2.getTrackbarPos('Vmin', 'color')
    Vmax = cv2.getTrackbarPos('Vmax', 'color')

    filtro_min = np.array([Hmin, Smin, Vmin], dtype=np.uint8)
    filtro_max = np.array([Hmax, Smax, Vmax], dtype=np.uint8)

    mask = cv2.inRange(hsv, filtro_min, filtro_max)

    cv2.imshow('color', mask)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break


cv2.destroyAllWindows()