import cv2
import numpy as np
import imutils
import time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
#cap.set(cv2.CAP_PROP_FPS, 5)
#fps = int(cap.get(5))


while True:
    start = time.time()
    _, img = cap.read()
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# COLORS
    # BLUE COLOR
    low_blue = np.array([90, 80, 2])
    high_blue = np.array([130, 255, 255])
    
    #GREEN COLOR
    low_green = np.array([35, 70, 80])
    high_green = np.array([70, 255, 255])
    
    # RED COLOR
    low_red = np.array([0, 50, 50])
    high_red = np.array([10, 255, 255])


# MASKS
    blue_mask = cv2.inRange(hsv_img, low_blue, high_blue)
    green_mask = cv2.inRange(hsv_img, low_green, high_green)
    red_mask = cv2.inRange(hsv_img, low_red, high_red)

   

# CONTOURS
    # BLUE contour
    ret, thresh = cv2.threshold(gray_img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    

# COLOR and SHAPE DETECTION
    # for blue object
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1600 and area < 16000:
            
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True),True)
            print(approx)
            cv2.drawContours(img, [contour], -1, (0, 255, 0),3)

            # Finding center point of shape
            O = cv2.moments(contour)
            if O['m00'] != 0.0:
                x = int(O['m10']/O['m00'])
                y = int(O['m10']/O['m00'])
            
        
            cv2.circle(img,(x,y),3,(255,255,255),-1)
            # cv2.putText(img, "BLUE", (x-20,y-20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255),3)

            # putting shape name at center of each shape
            if len(approx) == 3:
                cv2.putText(img, 'Triangle', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
  
            elif len(approx) == 4:
                cv2.putText(img, 'Square', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
  
            elif len(approx) > 100:
                cv2.putText(img, 'Circle', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    
    
    cv2.imshow("camera",img)

    stop = time.time()
    time.sleep(0.1)

    key = cv2.waitKey(1)
    if key == 27:
        break
    