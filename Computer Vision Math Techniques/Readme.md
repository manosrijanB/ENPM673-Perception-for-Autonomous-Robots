
## Problem Overview:

This homework consists of four problems that require applying various techniques in perception for autonomous robots. The key areas covered include camera field of view calculations, curve fitting, linear regression, and homography computation.

---
For more detailed  solutions, please refer to the [mbattula_hw1.pdf](https://github.com/manosrijanB/ENPM673-Perception-for-Autonomous-Robots/blob/48f679381babc1ccc124b95d18a8d3fdc03e7cc8/Homework-%201/mbattula_hw1.pdf)
 and code in the accompanying files.

## Problem 1: Field of View and Object Detection

**Objective:**  
- **Part 1:** Calculate the Field of View (FoV) of a camera using its resolution, sensor width, and focal length.
- **Part 2:** Compute the minimum number of pixels an object occupies in the image, given its size and distance from the camera.

**Solution:**  
- **FoV Calculation:** Using the formula:
``` math
  [
  \phi = 2 \times \tan^{-1}\left(\frac{d}{2f}\right)
  ]
```
  where \( f \) is the focal length and \( d \) is the sensor dimension.
- **Object Size:** With the given parameters, the height of the object in the image can be computed using the relation
 ``` math
  ( frac{h_0}{d_0} = \frac{h_f}{f} ).
``` 

---

## Problem 2: Curve Fitting for Ball Trajectory

**Objective:**  
- Track the trajectory of a ball using two video files, one without noise and one with noise, and fit curves using the least squares method.

**Steps:**  
1. Read the video files and extract the red ball from a white background by filtering the blue channel.
2. Track the ball using the detected topmost and bottommost pixels.
3. Fit a parabola to the data points from both videos.
4. Plot the best fit curves.

**Implementation Details:**
- **Video Processing:** Use OpenCV to read video frames and extract ball data points.
- **Curve Fitting:** The least squares method is used to fit a parabolic curve to the extracted data.

 
---

## Problem 3: Covariance Matrix, Eigenvalues, and Regression

**Objective:**  
- **Part 1:** Compute the covariance matrix from the provided data and compute its eigenvalues and eigenvectors.
- **Part 2:** Fit a line using the least squares method, total least squares, and RANSAC, and compare the results.

**Steps:**  
1. **Covariance and Eigenvalues:** Compute covariance and find the eigenvectors that describe the principal components of the data.
2. **Curve Fitting:** Compare linear regression techniques (Least Squares, Total Least Squares, and RANSAC) by fitting lines to the data points and visualizing the results.



---

## Problem 4: Homography and Singular Value Decomposition (SVD)

**Objective:**  
- Compute the homography matrix using SVD for given point correspondences.

**Steps:**  
1. **SVD Calculation:** Using matrix decomposition, compute the Singular Value Decomposition (SVD) of the matrix to find the homography.
2. **Matrix Computation:** Apply the derived equations to compute the homography matrix.

---
## Results:
### Standard Least Squares Curve Fitting
Perfect Data  |  Result |
:-------------------------:|:-------------------------:
<img src= "https://github.com/manosrijanB/ENPM673-Perception-for-Autonomous-Robots/blob/main/Computer%20Vision%20Math%20Techniques/output/without%20noise.png"> | <img src="https://github.com/manosrijanB/ENPM673-Perception-for-Autonomous-Robots/blob/main/Computer%20Vision%20Math%20Techniques/output/fitted%20without%20noise.png" width="100%"> 

Noisy Data   |  Result |
:-------------------------:|:-------------------------:
<img src= "https://github.com/manosrijanB/ENPM673-Perception-for-Autonomous-Robots/blob/main/Computer%20Vision%20Math%20Techniques/output/with%20noise.png"> | <img src="https://github.com/manosrijanB/ENPM673-Perception-for-Autonomous-Robots/blob/main/Computer%20Vision%20Math%20Techniques/output/fitted%20with%20noise.png" width="100%"> 

### LS, TLS, RANSAC, Covariance
Least Squares  |  Total Least Squares | Covariance |
:-------------------------:|:-------------------------:|:-------------------------:
<img src= "https://github.com/manosrijanB/ENPM673-Perception-for-Autonomous-Robots/blob/main/Computer%20Vision%20Math%20Techniques/output/lls.png"> | <img src="https://github.com/manosrijanB/ENPM673-Perception-for-Autonomous-Robots/blob/main/Computer%20Vision%20Math%20Techniques/output/TLS.png"> | <img src="https://github.com/manosrijanB/ENPM673-Perception-for-Autonomous-Robots/blob/main/Computer%20Vision%20Math%20Techniques/output/covariance.png" width="100%"> 

<p align="center">
  <img src="https://github.com/manosrijanB/ENPM673-Perception-for-Autonomous-Robots/blob/main/Computer%20Vision%20Math%20Techniques/output/ransac.png" width="50%">
</p>

### S, U, Vt Matrices (SVD)
S | U | Vt |
:-------------------------:|:-------------------------:|:-------------------------:
<img src= "https://github.com/manosrijanB/ENPM673-Perception-for-Autonomous-Robots/blob/main/Computer%20Vision%20Math%20Techniques/output/s%20matrix.png"> | <img src="https://github.com/manosrijanB/ENPM673-Perception-for-Autonomous-Robots/blob/main/Computer%20Vision%20Math%20Techniques/output/u%20matrix.png"> | <img src="https://github.com/manosrijanB/ENPM673-Perception-for-Autonomous-Robots/blob/main/Computer%20Vision%20Math%20Techniques/output/v%20matrix.png" width="100%"> 


---

## Instructions to Run:

1. Download all necessary files (e.g., **.csv data** for Problem 3).
2. Install required libraries such as **OpenCV** for video processing and **NumPy** for matrix operations.
3. **Problem 2:**
   - The program will prompt you to select between two video files: 
     - **Video 1:** No noise
     - **Video 2:** With noise
   - Enter the number corresponding to the video to see the curve fitting result.
   - If any other number is entered, the program will display "Video not available in directory" and terminate.

4. **Problem 3:**
   - The program will generate a graph containing data points, eigenvectors, and the fitted curves for each method (Least Squares, Total Least Squares, and RANSAC).

5. **Problem 4:**
   - The program uses SVD to compute the homography matrix. Python functions are defined to compute the SVD and homography matrix based on the provided correspondences.


## Dependencies:

- Python 3.x
- NumPy
- OpenCV





