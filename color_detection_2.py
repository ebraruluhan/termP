import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)


while True:
    _, img = cap.read()
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # BLUE COLOR
    low_blue = np.array([94, 80, 2])
    high_blue = np.array([126, 255, 255])
    blue_mask = cv2.inRange(hsv_img, low_blue, high_blue)
    blue = cv2.bitwise_and(img, img, mask=blue_mask)
    
    
    #GREEN COLOR
    low_green = np.array([25, 52, 72])
    high_green = np.array([102, 255, 255])
    green_mask = cv2.inRange(hsv_img, low_green, high_green)
    green = cv2.bitwise_and(img, img, mask=green_mask)
    

    # RED COLOR
    low_red = np.array([0, 50, 120])
    high_red = np.array([10, 255, 255])
    red_mask = cv2.inRange(hsv_img, low_red, high_red)
    red = cv2.bitwise_and(img, img, mask=red_mask)


    cv2.imshow("Camera", img)
    cv2.imshow("Red Mask", red)
    cv2.imshow("Green Mask", green)
    cv2.imshow("Blue Mask", blue)
    key = cv2.waitKey(1)
    if key == 27:
        break     