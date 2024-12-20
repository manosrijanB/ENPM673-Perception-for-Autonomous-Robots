# **Midterm Project - Image Processing Solutions**

This repository contains solutions for four image processing tasks completed as part of the ENPM673: Perception for Autonomous Robots course. The tasks involve image manipulation, feature extraction, camera calibration, and clustering, all implemented using Python and OpenCV.

---

## **Project Tasks**

### **1. Coin Extraction and Counting**
**Objective**: Separate coins in a binary image and count them using morphological operations and contour detection.  
**Pipeline**:
1. **Morphological Operations**: 
   - Apply **erosion** to shrink coin edges and separate overlapping coins.
   - Perform **dilation** to restore coin sizes post-separation.
2. **Edge Detection**: Convert the processed image to grayscale, blur it, and apply **Canny edge detection**.
3. **Contour Detection**: Use OpenCV’s `findContours` to detect and count the coins.
4. **Visualization**: Highlight each coin and display the total count.

**Results**:
- Successfully separated overlapping coins and counted them.
- Output:
  <p align="center">
    <img src="https://github.com/manosrijanB/ENPM673-Perception-for-Autonomous-Robots/blob/main/Image%20Processing%20and%20Camera%20Calibration/output/highlighted.png" width="50%">
    <img src="https://github.com/manosrijanB/ENPM673-Perception-for-Autonomous-Robots/blob/main/Image%20Processing%20and%20Camera%20Calibration/output/counted.png" width="50%">
  </p>

---

### **2. Image Stitching**
**Objective**: Stitch two images into a single panorama using feature matching and homography.  
**Pipeline**:
1. **Feature Detection**:
   - Use **ORB (Oriented FAST and Rotated BRIEF)** to detect keypoints and descriptors.
2. **Feature Matching**:
   - Apply **BFMatcher** with **KNN matching** to find corresponding points between images.
3. **Homography Transformation**:
   - Compute the homography matrix to align the images.
4. **Stitching**:
   - Warp one image onto the other using the homography matrix.
5. **Post-Processing**:
   - Remove stitching artifacts using **median blur**.

**Results**:
- Accurate stitching with seamless blending of images.
- Output:
  <p align="center">
    <img src="https://github.com/manosrijanB/ENPM673-Perception-for-Autonomous-Robots/blob/main/Image%20Processing%20and%20Camera%20Calibration/output/stiched%20without%20noise.png" width="50%">
  </p>

---

### **3. Camera Calibration**
**Objective**: Compute the intrinsic camera matrix \( K \) using world-to-image point correspondences.  
**Pipeline**:
1. **Formulate Equations**:
   - Use the relationship \( x = PX \), where \( P = [M|C] \), and solve for the projection matrix \( P \).
2. **Singular Value Decomposition (SVD)**:
   - Solve the homogeneous linear system \( AP = 0 \) using SVD to minimize errors.
3. **Extract Intrinsic Matrix**:
   - Decompose \( M = KR \), where \( K \) is upper triangular (intrinsic matrix) and \( R \) is orthogonal (rotation matrix).
4. **Normalize**:
   - Ensure the intrinsic matrix values are properly scaled.

**Results**:
- Successfully computed the intrinsic matrix using only NumPy.
- Output:
  <p align="center">
    <img src="https://github.com/manosrijanB/ENPM673-Perception-for-Autonomous-Robots/blob/main/Image%20Processing%20and%20Camera%20Calibration/output/intrinsic%20camera%20matrix.png" width="50%">
  </p>

---

### **4. K-means Image Segmentation**
**Objective**: Segment an image into four color classes using a custom implementation of the K-means clustering algorithm.  
**Pipeline**:
1. **Preprocessing**:
   - Flatten the image into a 2D array of pixel intensities.
2. **Centroid Initialization**:
   - Randomly initialize cluster centroids.
3. **Clustering**:
   - Assign each pixel to the nearest centroid based on Euclidean distance.
   - Update centroids iteratively by averaging cluster members.
4. **Convergence**:
   - Repeat until centroids stabilize.
5. **Visualization**:
   - Recreate the image with clustered colors.

**Results**:
- Segmented the image into distinct regions based on color similarity.
- Output:
  <p align="center">
    <img src="https://github.com/manosrijanB/ENPM673-Perception-for-Autonomous-Robots/blob/main/Image%20Processing%20and%20Camera%20Calibration/Figure_1.png" width="50%">
  </p>

---

## **Directory Structure**
```bash
├── Code
│   ├── question1.py  # Coin extraction and counting
│   ├── question2.py  # Image stitching
│   ├── question3.py  # Camera calibration
│   ├── question4.py  # K-means image segmentation
├── Output
│   ├── highlighted.png
│   ├── counted.png
│   ├── stitched_without_noise.png
│   ├── intrinsic_camera_matrix.png
│   ├── Figure_1.png
├── README.md
├── mbattula_midterm.pdf  # Report with solutions and explanations
```

---

## **How to Run**

### **Dependencies**
Install the required Python packages:
```bash
pip install numpy opencv-python matplotlib
```

### **Execution**
Run individual scripts for each task:
1. **Coin Extraction and Counting**:
   ```bash
   python question1.py
   ```
2. **Image Stitching**:
   ```bash
   python question2.py
   ```
3. **Camera Calibration**:
   ```bash
   python question3.py
   ```
4. **K-means Image Segmentation**:
   ```bash
   python question4.py
   ```

---

## **Acknowledgments**
This project was completed as part of the **ENPM673: Perception for Autonomous Robots** course. Special thanks to the course instructors and teaching assistants for their guidance.

For more details, refer to the [Midterm Report](mbattula_midterm.pdf).
