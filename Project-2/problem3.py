import cv2 as cv
import numpy as np
import math 

def filtering_colors(image):
	# Filter white pixels

#  for lower white  
	lower_white = np.array([220, 220, 220])
	upper_white = np.array([255, 255, 255])
	white_mask = cv.inRange(image, lower_white, upper_white)
	white_image = cv.bitwise_and(image, image, mask=white_mask)
	# Filter yellow pixels
	hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
	lower_yel = np.array([10,90,80])
	upper_yel = np.array([110,255,255])
	yel_mask = cv.inRange(hsv, lower_yel, upper_yel)
	yel_image = cv.bitwise_and(image, image, mask=yel_mask)
	# Combine the two above images
	image2 = cv.addWeighted(white_image, 1., yel_image, 1., 0.)
	return image2




poly_left = None
poly_right = None

video = cv.VideoCapture("challenge.mp4")

fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('predict_turn.avi', fourcc, 30.0, (960,540))
while True:
    isTrue,frame = video.read()
    # frame = frame.copy()
    filter = filtering_colors(frame)
   
    gray = cv.cvtColor(filter , cv.COLOR_BGR2GRAY)
    poly = gray.copy()
    # points of interest
    inter_poly = np.array( [[[0,720],[200,700],[600,400],[780,420],[1240,710],[1280,720],[1280,0],[0,0]]],dtype=np.int32)
    cv.fillPoly( poly , inter_poly, (0,0,0) )
    blurred =  cv.GaussianBlur(poly,(11,11),cv.BORDER_DEFAULT)
    

    retval,threshed = cv.threshold(blurred,150,250,cv.THRESH_BINARY)

    
    cv.imshow("threshed",threshed)
    
    h,w = frame.shape[:2]
    cornerpoints = [ (600,424),(780,424),(1100,660),(240,660)]
  
    k1 = np.float32([[(0,0),(200,0),(200,200),(0,200)]])
    k2=np.float32(cornerpoints)
    h1, w1,s = frame.shape 
    H = cv.getPerspectiveTransform(k2,k1)
   
    Hinv = cv.getPerspectiveTransform(k1,k2)
    
    
    
    filter_frame = filtering_colors(frame)
    
    warp = cv.warpPerspective(filter_frame,H,(200,200))
    warp = cv.cvtColor(warp,cv.COLOR_BGR2GRAY)
    warp= cv.GaussianBlur(warp,(3,3), cv.BORDER_DEFAULT)
    # warp= cv.erode(warp, (5,5), iterations=1)
    # warp= cv.dilate(warp, (5,5), iterations=5)
            
    retval,threshed_warp = cv.threshold(warp,150,255,cv.THRESH_BINARY)
    cv.imshow("bird's eye",threshed_warp)
    
    
    leftx, poly_left = np.where(threshed_warp[:, :100] == 255)
    coef = np.polyfit(leftx, poly_left, 2)
    leftx = np.arange(70, 200, 1)
    poly_left = np.polyval(coef, leftx)
    
    if poly_left is None: poly_left = poly_left
    poly_left = 0.8*poly_left + 0.2*poly_left
    
    leftpt = np.int0(np.c_[poly_left, leftx])
    
    rightx, righty = np.where(threshed_warp[:, 130:] == 255)
    coef = np.polyfit(rightx, righty, 3)
    rightx = np.arange(70, 200, 1)
    righty = np.polyval(coef, leftx)
    
    if poly_right is None: poly_right = righty
    poly_right = 0.8*poly_right + 0.2*righty
    
    rightpt = np.int0(np.c_[130 + poly_right, rightx])
    
    new = np.uint8(np.zeros((200, 200, 3))*255)
    cv.polylines(new, [leftpt], False, [0, 0, 255], 4)
    cv.polylines(new, [rightpt], False, [0, 0, 255], 4)
    
    pts = np.r_[leftpt, np.flipud(rightpt)]
    cv.fillPoly(new, [pts], [0,220, 0])
    
    lanes1 = cv.warpPerspective(new, Hinv, (w, h), flags = cv.INTER_LINEAR)
    over = np.uint8(0.5*frame.copy() + 0.3*lanes1)
    # result = show_curvatures(over, left_fit, right_fit, 3/1280, 30/720)
    cv.imshow('filtered', filter_frame)
    cv.imshow("final",over)
    out.write(over) 
    if cv.waitKey(20) & 0xFF==ord("k"):
        break
video.release()
out.release()
cv.destroyAllWindows()