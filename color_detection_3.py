import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

while True:
    _, img = cap.read()
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

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
    blue_mask = cv2.inRange(hsv_img, low_blue, high_blue)
    green_mask = cv2.inRange(hsv_img, low_green, high_green)
    red_mask = cv2.inRange(hsv_img, low_red, high_red)

# CONTOURS
    # BLUE contour
    blue_contours = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    blue_contours = imutils.grab_contours(blue_contours)
    
    # GREEN contour
    green_contours = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    green_contours = imutils.grab_contours(green_contours)

    # RED contour
    red_contours = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    red_contours = imutils.grab_contours(red_contours)


# COLOR DETECTION
    # for blue object
    for contour in blue_contours:
        blue_area = cv2.contourArea(contour)
        if blue_area>5000:
            
            cv2.drawContours(img, [contour], -1, (0, 255, 0),3)
            
            # Finding center point of shape
            O = cv2.moments(contour)
            if O['m00'] != 0.0:
                x = int(O['m10']/O['m00'])
                y = int(O['m10']/O['m00'])
            
        
            cv2.circle(img,(x,y),7,(255,255,255),-1)
            cv2.putText(img, "BLUE", (x-20,y-20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255),3)

    # for green object
    for contour in green_contours:
        green_area = cv2.contourArea(contour)
        if green_area>5000:
            
            cv2.drawContours(img, [contour], -1, (0, 255, 0),3)
            
            # Finding center point of shape
            O = cv2.moments(contour)
            if O['m00'] != 0.0:
                x = int(O['m10']/O['m00'])
                y = int(O['m10']/O['m00'])
            
        
            cv2.circle(img,(x,y),7,(255,255,255),-1)
            cv2.putText(img, "GREEN", (x-20,y-20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255),3)

    # for red object
    for contour in red_contours:
        red_area = cv2.contourArea(contour)
        if red_area>5000:
            
            cv2.drawContours(img, [contour], -1, (0, 255, 0),3)
            
            # Finding center point of shape
            O = cv2.moments(contour)
            if O['m00'] != 0.0:
                x = int(O['m10']/O['m00'])
                y = int(O['m10']/O['m00'])
            
        
            cv2.circle(img,(x,y),7,(255,255,255),-1)
            cv2.putText(img, "RED", (x-20,y-20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255),3)

            

    cv2.imshow("camera",img)

    key = cv2.waitKey(1)
    if key == 27:
        break
    
