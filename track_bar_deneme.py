import cv2
import numpy as np
from matplotlib import pyplot as plt


cap = cv2.VideoCapture(0)

def nothing(x):
    pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L-H","Trackbars",0,180,nothing)
cv2.createTrackbar("L-S","Trackbars",0,255,nothing)
cv2.createTrackbar("L-V","Trackbars",0,255,nothing)
cv2.createTrackbar("U-H","Trackbars",180,180,nothing)
cv2.createTrackbar("U-S","Trackbars",255,255,nothing)
cv2.createTrackbar("U-V","Trackbars",255,255,nothing)

while True:
    _, img = cap.read()
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # COLORS
    # BLUE COLOR
    low_blue = np.array([90, 80, 0])
    high_blue = np.array([120, 255, 255])
    
    #GREEN COLOR
    low_green = np.array([35, 70, 80])
    high_green = np.array([70, 255, 255])
    
    # RED COLOR
    low_red = np.array([0, 50, 120])
    high_red = np.array([10, 255, 255])
   
   # MASKS
    blue_mask = cv2.inRange(img_hsv, low_blue, high_blue)
    green_mask = cv2.inRange(img_hsv, low_green, high_green)
    red_mask = cv2.inRange(img_hsv, low_red, high_red)
   
    cv2.imshow('Result', img)
    cv2.imshow("Blue Mask", blue_mask)
    cv2.imshow("Green Mask", green_mask)
    cv2.imshow("Red Mask", red_mask)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()