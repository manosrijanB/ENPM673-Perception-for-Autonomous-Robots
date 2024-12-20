# Main Repository ReadMe: Perception for Autonomous Robots Projects

This repository serves as a comprehensive collection of projects related to computer vision and perception, focusing on topics like motion segmentation, stereo vision, AR tag detection, image stitching, and lane detection. These projects demonstrate the application of advanced image processing and machine learning techniques in autonomous robotics.

---

## **Project Highlights**

### **1. Motion Segmentation Using Optical Flow**
- **Objective**: Perform motion segmentation to differentiate between static and moving objects using the RAFT optical flow algorithm.
- **Key Features**:
  - Probabilistic model-based segmentation inspired by Bideau et al.'s "It’s Moving!"
  - Dataset: KITTI MOTS.
  - Technologies: Python, MATLAB, RAFT Optical Flow.
- **Results**:
  - Short video segments successfully segmented into static and dynamic objects.

[Explore Detailed ReadMe](Optical flow based motion segmentation\README.md)

---

### **2. Stereo Vision and Depth Estimation**
- **Objective**: Implement a stereo vision system for disparity and depth estimation using calibrated monocular images.
- **Key Features**:
  - Rectification of epipolar lines for precise disparity computation.
  - Block-matching for disparity map generation and depth computation.
  - Tools: Python, OpenCV, NumPy.
- **Results**:
  - Accurate disparity and depth maps produced for stereo images.


[Explore Detailed ReadMe](https://github.com/manosrijanB/ENPM673-Perception-for-Autonomous-Robots/blob/main/Optical%20flow%20based%20motion%20segmentation/README.md)

---

### **3. AR Tag Detection and Tracking**
- **Objective**: Detect and track AR tags in video sequences while superimposing images and projecting 3D objects.
- **Key Features**:
  - Techniques: Homography, FFT, Harris Corner Detection.
  - 3D Cube Projection: Overlay virtual 3D cubes onto AR tags.
  - Tools: Python, OpenCV.
- **Results**:
  - Successfully decoded and replaced AR tags in real-time video.

[Explore Detailed ReadMe](AR_Tag_Detection/README.md)

---

### **4. Image Processing and Camera Calibration**
- **Objective**: Solve image processing problems like coin extraction, image stitching, camera calibration, and K-means segmentation.
- **Key Features**:
  - Camera calibration using intrinsic matrix estimation.
  - Morphological operations for object segmentation.
  - Feature matching for panorama stitching.
- **Results**:
  - Accurate calibration, segmentation, and stitched images.

[Explore Detailed ReadMe](Image_Processing_Calibration/README.md)

---

### **5. Lane Detection and Turn Prediction**
- **Objective**: Detect lanes and predict road turns based on video sequences.
- **Key Features**:
  - Histogram equalization for contrast enhancement.
  - Lane classification (solid vs. dashed) using slope analysis.
  - Turn prediction with curvature radius estimation.
- **Results**:
  - Reliable detection and prediction under varying road conditions.

[Explore Detailed ReadMe](Lane_Detection_Turn_Prediction/README.md)



## **How to Use**
1. **Install Required Dependencies**:
   ```bash
   pip install numpy opencv-python matplotlib scipy torch torchvision
   ```
2. **Navigate to Project Directories**:
   Each project contains its own set of instructions and scripts.
   ```bash
   cd Project_Directory
   python script_name.py
   ```
3. **View Outputs**:
   Results for each project are saved in their respective `results/` or `outputs/` folders.

---

## **Acknowledgments**
This repository consolidates projects completed as part of the **ENPM673: Perception for Autonomous Robots** course. Special thanks to the instructors and teaching assistants for their guidance.

For more details, refer to the detailed README files or reports in each project folder.
