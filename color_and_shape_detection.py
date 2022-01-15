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
    img = img[100:500, 100:500,:]
    img = cv2.flip(img, 1)
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
    blue_contours = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    blue_contours = imutils.grab_contours(blue_contours)
    
    # GREEN contour
    green_contours = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    green_contours = imutils.grab_contours(green_contours)

    # RED contour
    red_contours = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    red_contours = imutils.grab_contours(red_contours)


# COLOR and SHAPE DETECTION
    # for blue object
    for contour in blue_contours:
        blue_area = cv2.contourArea(contour)
        if blue_area>100:
            
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True),True)
            cv2.drawContours(img, [contour], -1, (0, 255, 0),3)

            # Finding center point of shape
            O = cv2.moments(contour)
            if O['m00'] != 0.0:
                x = int(O['m10']/O['m00'])
                y = int(O['m10']/O['m00'])
            
        
            cv2.circle(img,(x,y),7,(255,255,255),-1)
            cv2.putText(img, "BLUE", (x-20,y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),2)

            # putting shape name at center of each shape
            if len(approx) == 3:
                cv2.putText(img, 'Triangle', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
  
            elif len(approx) == 4:
                cv2.putText(img, 'Square', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            elif len(approx) >= 100:   
                cv2.putText(img, 'Circle', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # for green object
    for contour in green_contours:
        green_area = cv2.contourArea(contour)
        if green_area>100:
            
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True),True)
            cv2.drawContours(img, [contour], -1, (0, 255, 0),3)
            
            # Finding center point of shape
            O = cv2.moments(contour)
            if O['m00'] != 0.0:
                x = int(O['m10']/O['m00'])
                y = int(O['m10']/O['m00'])
            
        
            cv2.circle(img,(x,y),7,(255,255,255),-1)
            cv2.putText(img, "GREEN", (x-20,y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),2)

            # putting shape name at center of each shape
            if len(approx) == 3:
                cv2.putText(img, 'Triangle', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
  
            elif len(approx) == 4:
                cv2.putText(img, 'Square', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
  
            elif len(approx) >= 100:
                cv2.putText(img, 'Circle', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # for red object
    for contour in red_contours:
        red_area = cv2.contourArea(contour)
        if red_area>100:
            
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True),True)
            cv2.drawContours(img, [contour], -1, (0, 255, 0),3)
            
            # Finding center point of shape
            O = cv2.moments(contour)
            if O['m00'] != 0.0:
                x = int(O['m10']/O['m00'])
                y = int(O['m10']/O['m00'])
            
        
            cv2.circle(img,(x,y),7,(255,255,255),-1)
            cv2.putText(img, "RED", (x-20,y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),2)

            # putting shape name at center of each shape
            if len(approx) == 3:
                cv2.putText(img, 'Triangle', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
  
            elif len(approx) == 4:
                cv2.putText(img, 'Square', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
  
            elif  len(approx) >= 100:
                cv2.putText(img, 'Circle', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    
    cv2.imshow("camera",img)

    stop = time.time()
    time.sleep(0.1)

    print(stop-start)

    key = cv2.waitKey(1)
    if key == 27:
        break
    