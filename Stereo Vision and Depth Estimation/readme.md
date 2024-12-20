# Stereo Vision and Depth Estimation

# Introduction:
In this project, we implemented a stereo vision system to compute disparity and depth images. We also estimated the camera poses with respect to a given scene using 6 images from a monocular camera and their corresponding feature points. The process involves several key steps: calibration, rectification, correspondence matching, and depth estimation.

# Required Modules:
- numpy
- matplotlib
- cv2 (OpenCV)
- tqdm

Ensure that all dependencies are installed. You can use the following command to install them:



# Steps Involved:

### Calibration:

Match features in two images using SIFT and Brute-Force Matcher.
Estimate the Fundamental Matrix and use RANSAC to filter outliers.
Calculate the Essential Matrix and decompose it to obtain the camera poses.

### Rectification:

Apply perspective transformations to align epipolar lines horizontally.
Compute the homography matrices (H1, H2) and rectify the images.
Visualize the epipolar lines on both the original and rectified images.

### Correspondence Matching:

<p align="center">
  <img src="https://github.com/manosrijanB/ENPM673-Perception-for-Autonomous-Robots/blob/main/Stereo%20Vision%20and%20Depth%20Estimation/unrectified%20epipolar.png" width="50%">
</p>

<p align="center">
  <img src="https://github.com/manosrijanB/ENPM673-Perception-for-Autonomous-Robots/blob/main/Stereo%20Vision%20and%20Depth%20Estimation/rectified%20epiroal%20lines.png" width="50%">
</p>
![unrectified epipolar](Stereo Vision System\unrectified epipolar.png)
![rectified epiroal lines](Stereo Vision System\rectified epiroal lines.png)

### Disparity:
Use block matching techniques (SSD) to find corresponding points between the rectified images.
Calculate the disparity map and rescale it for visualization.

<p align="center">
  <img src="https://github.com/manosrijanB/ENPM673-Perception-for-Autonomous-Robots/blob/main/Stereo%20Vision%20and%20Depth%20Estimation/disparity(grey).png" width="50%">
</p>

<p align="center">
  <img src="https://github.com/manosrijanB/ENPM673-Perception-for-Autonomous-Robots/blob/main/Stereo%20Vision%20and%20Depth%20Estimation/disparity%20(hot).png" width="50%">
</p>


## Depth Image Computation



Use the disparity information to compute the depth map for each pixel in the image.
Visualize the depth map as both grayscale and color images.

<p align="center">
  <img src="https://github.com/manosrijanB/ENPM673-Perception-for-Autonomous-Robots/blob/main/Stereo%20Vision%20and%20Depth%20Estimation/depth%20map(gray%20scale).png" width="50%">
</p>

<p align="center">
  <img src="https://github.com/manosrijanB/ENPM673-Perception-for-Autonomous-Robots/blob/main/Stereo%20Vision%20and%20Depth%20Estimation/depth%20map%20(hot).png" width="50%">
</p>

## How to Run the Project:
Ensure the Dataset is in the Correct Folder

Keep the working directory as the current folder where the project files and dataset folders are located.
### Running the Script:

Execute the project3.py file to compute the required outputs.
Output Results

All outputs, including disparity and depth maps, will be saved in the outputs folder.

## Dataset Folders

Ensure that the dataset folders (containing the stereo images and corresponding data) are present in the current directory.

## Final Notes:
The project relies on accurate feature matching and calibration to compute the disparity and depth maps.
If you encounter issues with the disparity map or depth image, ensure that the rectification step was performed correctly and that the feature matching is accurate.




