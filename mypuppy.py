import cv2
import numpy as np


img = cv2.imread('C:/Users/uluha/Desktop/Computer-Vision-with-Python/DATA/shapes2.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## THRESHOLDING PART
ret , threshold = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)

## FINDING CONTOURS
image,contours,hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

## NAMES of GIVEN SHAPES
i = 0

for contour in contours:
  
    if i == 0:
        i = 1
        continue
  
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
      
    cv2.drawContours(img, [contour], 0, (0, 255, 0), 5)
  
    # finding center point of shape
    O = cv2.moments(contour)
    if O['m00'] != 0.0:
        x = int(O['m10']/O['m00'])
        y = int(O['m01']/O['m00'])
  
    # putting shape name at center of each shape
    if len(approx) == 3:
        cv2.putText(img, 'Triangle', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
  
    elif len(approx) == 4:
        cv2.putText(img, 'Square', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
  
    elif len(approx) == 5:
        cv2.putText(img, 'Pentagon', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
  
    elif len(approx) == 6:
        cv2.putText(img, 'Hexagon', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
  
    else:
        cv2.putText(img, 'Circle', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
  

cv2.imshow('shapes', img)  
cv2.waitKey(0)
cv2.destroyAllWindows()