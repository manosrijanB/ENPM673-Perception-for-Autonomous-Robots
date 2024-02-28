

import cv2
import numpy as np
import matplotlib.pyplot as plt
# reading the video file
cap = cv2.VideoCapture("/home//chaosmachete/Desktop/ball_video2.mp4")
x=[] 
y=[]
#  loop
while(cap.isOpened()):
    # extracting the frames
    ret, frame = cap.read()
    if not ret : break

    # print(frame.shape)
    #  extarcting blue channel for the ball as it as the least pixels
    blue = frame[:, :, 0]
    print(np.min(blue))
    indices = np.where(blue <= 100)
    #  thresholding the blue pixels
    r, c = indices
    avgr = ((max(r)+min(r))/ 2)
    avgr=1700-avgr
    # locating the center
    avgc = ((max(c)+min(c))/ 2)

    x.append(avgr)
    y.append(avgc)
plt.scatter(y,x)
x,y = y, x
x = np.array(x)
y = np.array(y)

E = np.vstack([x**2, x, np.ones(len(x))]).T 
 
# linear least sqaure 
m = np.linalg.inv(E.T@ E) @E.T @ y 

#  new fitted y  
newy = E @ m 
plt.plot(x, newy, 'y')
plt.show()


            


    




