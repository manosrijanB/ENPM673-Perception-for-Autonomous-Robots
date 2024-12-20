import cv2 as cv
import numpy as np
import glob
from pathlib import Path
import matplotlib.pyplot as plt

def equi(img):
    hist, _ = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    
    cdf = cdf/cdf.max()
    w, h = img.shape
    for i in range(w):
        for j in range(h):
            inst = img[i, j]
            img[i, j] = cdf[inst]*255
   
    return img
def adaptive_equi(img, n = 8):
    img = img.copy()
    h, w = img.shape
    sh, sw = h//n, w//n
    for i in range(0, h, sh):
        for j in range(0, w, sw):
            img[i:i+sh, j:j+sw] = equi(img[i:i+sh, j:j+sw])
    return img



vid = cv.VideoCapture('output.avi')
fourcc = cv.VideoWriter_fourcc(*'XVID')
out1 = cv.VideoWriter('equilized.avi', fourcc, 25.0, (1224,370))
out2 = cv.VideoWriter('adaptiveequilized.avi', fourcc, 25.0, (1224,370))


while True:
    
 isTrue, frame = vid.read()

 img1 = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
 l, a, b = cv.split(img1)

 equilized = equi(l)
 adp_equilized = adaptive_equi(l)

 m1 = cv.merge([equilized, a, b])
 m2 = cv.merge([adp_equilized, a, b])
#  clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
 
 m1 = cv.cvtColor(m1, cv.COLOR_LAB2BGR)
 m2 = cv.cvtColor(m2, cv.COLOR_LAB2BGR)

 cv.imshow('orginal', frame)
 
 cv.imshow('equilized', m1)
 cv.imshow('adaptive equilized', m2)
 out1.write(m1)
 out2.write(m2)
 
 if cv.waitKey(20) & 0xFF==ord("q"):
      break 
out1.release()  
out2.release()  
