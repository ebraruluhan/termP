import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


while True:
    _, img = cap.read()
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    height, width, _ = img.shape

    cx = int(width / 2)
    cy = int(height / 2)

    pixel_center = hsv_img[cy,cx]

    h_value = pixel_center[0]
    
    color = 'Undefined'
    if h_value < 5:
        color = "RED"
    elif h_value < 78:
        color = "GREEN"
    elif h_value < 131:
        color = "BLUE"

    print(pixel_center)
    cv2.putText(img, color, (10,50), 2, 1, (0, 0, 255), 2)
    cv2.circle(img, (cx,cy), 5, (0,255,0), 3)




    cv2.imshow("camera",img)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows
