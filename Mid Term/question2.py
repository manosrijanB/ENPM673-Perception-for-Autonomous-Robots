import cv2 as cv

import numpy as np

# reading of the images
imgA = cv.imread('Q2imageA.png')
imgB = cv.imread('Q2imageB.png')

#  using the orb as it is efficient to track features
orb_obj = cv.ORB_create()
kp1 = orb_obj.detect(imgA,None)
kp2 = orb_obj.detect(imgB,None)

# find the keypoints and descriptors with orb
kp1, des1 = orb_obj.compute(imgA, kp1)
kp2, des2 = orb_obj.compute(imgB, kp2)

# # BFMatcher with default parameters
bf = cv.BFMatcher()
#  finding matches between two images
matches = bf.knnMatch(des1, des2, k = 2)

# finding good matches with ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# stiching the two images
img3 = cv.drawMatchesKnn(imgA, kp1, imgB, kp2, good, flags = 2, outImg = None)

# source and desitination points 
sPoints = np.array([kp1[m[0].queryIdx].pt for m in good])
dPoints = np.array([kp2[m[0].trainIdx].pt for m in good])

# applying homography 
H, __ = cv.findHomography( dPoints, sPoints, cv.RANSAC, 4.0)
# assigning w and h for image a for warping
w, h,_ = imgA.shape
# for warping image b to image a
img_B_corners = np.array([[0, 0, 1], [0, w, 1], [h, w, 1], [h, 0, 1]])
warped_corners = H @ img_B_corners.T
warped_corners = np.int0(np.round(warped_corners/warped_corners[2]))

print(warped_corners)
h2 = warped_corners[0, -1]
warped_image = cv.warpPerspective(imgB, H, (h2, w))

#  creating a black image which will be superimposesd on image a
imgblack = np.zeros((w, h2, 3), np.uint8)
imgblack[:, :h , :] = imgA
cv.fillPoly(imgblack, [warped_corners[:2, :].T], 0)

# stiching the images
stiched = warped_image + imgblack
# clearing the unwanted line using median blur
# stiched = cv.medianBlur(stiched, ksize=3)
# showing the images
cv.imshow('matching lines', img3)
cv.imshow('stiched', stiched)
cv.waitKey(0)
