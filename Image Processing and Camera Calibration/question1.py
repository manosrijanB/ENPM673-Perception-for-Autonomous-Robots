import numpy as np
import cv2 as cv


# reading the image
img= cv.imread("Q1image .png") 


# creating a shape of circle kernal
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (45, 45))

# eroding and dilating the image to remove the bridges
img_erosion = cv.erode(img, kernel, iterations=1)
d = cv.dilate(img_erosion, kernel, iterations=1)

# converting into gray scale and blurring the image
gray = cv.cvtColor(d, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (3, 3), 0)
canny = cv.Canny(blur, 30, 150)

# for counting of the coins
(cnt, hierarchy) = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
rgb = cv.cvtColor(blur, cv.COLOR_BGR2RGB)
# to draw red colour coins in the output image
cv.drawContours(rgb, cnt, -1, (0, 0, 250), 2)

# printing of the number of coins
print("coins in the image : ", len(cnt))

# showing of all the images
cv.imshow('Input', img)
cv.imshow('Erosion', img_erosion)
cv.imshow('Dilation', d)
cv.imshow('coins', rgb)

cv.waitKey(0)    