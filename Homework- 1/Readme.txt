#Problem 2:

The program woudl ask you to enter the number for video file, 1 being for video 1 (without noise) and 2 being for video 2 (with noise). You get the curve for specified video as soon as you enter the required number. Entering any other number would result the statement "Video not available in directory" after which the program terminates.
I extracted red ball from white background as the red ball as least blue pixels i tracked the ball thresholding in less than 100 in BGR and to detect the top and bottom most pixels and took an avg for to compute center and thus detect the trajectory. 
 
#Problem 3:

The final output would be a graph containing data points, eigen vectors, and the curves obtained from least square method, total least square method and RANSAC method. 

#Problem 4:

To compute the homography, first we need to compute SVD. 
I have defined a function for SVD and used it to compute Homography matrix.
