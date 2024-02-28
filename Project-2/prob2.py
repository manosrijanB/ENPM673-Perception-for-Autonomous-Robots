import cv2 as cv
from cv2 import Canny
from cv2 import imshow
import numpy as np


vid = cv.VideoCapture('whiteline.mp4')

def draw_line(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(blank_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
# adding the two images
    img = cv.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img


fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('Lane Detection.avi', fourcc, 30.0, (960,540))
while True:
     isTrue, frame = vid.read()
#  converting to grayscale
     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #  to remove noise
     blur = cv.GaussianBlur(gray,(11,11),0)
        #to detect the edges  
     Canny = cv.Canny(blur, 50, 150)
    #  creating a copy 
     poly = Canny.copy()
    #  creating a mask
     mask = np.zeros_like(poly)
    #  vertices where the lanes lay 
     region_of_interest = np.array([[
        (100, 540), (450, 330), (520,330), (950,540)]], dtype=np.int32)
    # color of the mask which will be imposed
     match_mask_color = (255)
    #  creating a mask 
     cv.fillPoly(mask, region_of_interest, match_mask_color)
    #  adding the mask and canny image
     poly = cv.bitwise_and(poly, mask)
       
     cv,imshow('masked', poly)     
    
     retval, thresh = cv.threshold(poly, 150,200,cv.THRESH_BINARY)  
    # using hough lines algo to detect lines  
             
     lines = cv.HoughLinesP(thresh, rho=2, theta=np.pi/180,
                            threshold=10, lines=None, minLineLength=2, maxLineGap=10)

     img = np.copy(frame)
     img = draw_line(img, lines)

     img_shape = img.shape[0:2]
    
     middle_x = img_shape[1] / 2
        
     left_lane_lines = []
     right_lane_lines = []

     for line in lines:
            for x1, y1, x2, y2 in line:
                dx = x2 - x1 
                if dx == 0:
                    #Discarding line since we can't gradient is undefined at this dx
                    continue
                dy = y2 - y1
                
                # Similarly, if the y value remains constant as x increases, discard line
                if dy == 0:
                    continue
                
                slope = dy / dx
                epsilon = 0.3
                if abs(slope) <= epsilon:
                    continue
               
                elif x1 >= middle_x and x2 >= middle_x:
                    # Lane should also be within the right hand side of region of interest
                    right_lane_lines.append([[x1, y1, x2, y2]])
                    cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        
        
    #  right_colored_img = draw_line(img, right_lane_lines)
        

     out.write(img) 
    
     cv.imshow("orignal", frame)
     cv.imshow("gray scale", gray)
     cv.imshow('lane', img)
     
     out.write(img) 
     if cv.waitKey(200) & 0xFF==ord("q"):
      break 
out.release()
