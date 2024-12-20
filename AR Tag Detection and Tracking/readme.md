
---

# **AR Tag Detection and Tracking**

This repository contains the implementation for detecting and tracking Augmented Reality (AR) tags in a video sequence. The project focuses on identifying AR tags, decoding their IDs and orientation, superimposing images, and projecting 3D objects (a cube) onto the tags. These methods simulate core functionalities used in augmented reality systems.

---
<p align="center">
  <img src="https://github.com/manosrijanB/ENPM673-Perception-for-Autonomous-Robots/blob/main/AR%20Tag%20Detection%20and%20Tracking/3D%20projection1.gif" width="50%">
</p>

## **Project Overview**

The main objectives of this project are:
1. **AR Tag Detection**:
   - Identify the AR tag in video frames using image processing techniques like thresholding, Fourier Transforms, and corner detection.
2. **Image Superimposition**:
   - Replace the AR tag with a custom image (e.g., Testudo logo) using homography.
3. **3D Cube Projection**:
   - Render a virtual 3D cube on the AR tag by calculating a projection matrix and utilizing camera parameters.
4. **AR Tag Decoding**:
   - Decode the orientation and ID of the AR tag using its binary encoding scheme.

---

## **Implementation Details**

### **1. AR Tag Detection**
- **Techniques Used**:
  - Fast Fourier Transform (FFT) to enhance edges and filter noise.
  - Harris Corner Detection to extract corner points of the AR tag.
  - Thresholding and Morphological Operations for better tag isolation.
- **Output**:
  - Detected AR tag corners are marked on the video frames.

---

### **2. Image Superimposition**
- **Objective**:
  Replace the AR tag with a custom image (`testudo.png`).
- **Steps**:
  - Compute a **Homography Matrix** using corner correspondences.
  - Use a custom warping function to overlay the image onto the AR tag.
  - Display the superimposed image on each frame.

---

### **3. 3D Cube Projection**
- **Objective**:
  Overlay a 3D virtual cube onto the AR tag, maintaining its orientation and perspective.
- **Steps**:
  - Calculate the **Projection Matrix** using the Homography and camera intrinsic parameters.
  - Transform the 3D cube’s vertices to 2D image coordinates.
  - Connect the projected vertices to render the cube.

---

### **4. AR Tag Decoding**
- **Objective**:
  Decode the ID and orientation of the AR tag using its 4x4 binary grid encoding.
- **Steps**:
  - Extract the inner grid of the AR tag and determine the tag’s orientation.
  - Decode the 2x2 binary ID based on the white square’s position.
  - Rotate the grid for alignment.

---

## **Project Structure**

```plaintext
├── code/
│   ├── ar_tag_detection.py      # Core implementation
├── data/
│   ├── testudo.png              # Custom image for superimposition
│   ├── output.avi               # Input video containing AR tags
├── outputs/
│   ├── superimposed.avi         # Video with Testudo superimposed
│   ├── cube_projection.avi      # Video with 3D cube projection
├── README.md                    # Project documentation
```

---

## **How to Run**

### **Dependencies**
Install the required Python libraries:
```bash
pip install numpy opencv-python matplotlib
```

### **Steps to Execute**
1. Place the input video (`output.avi`) and custom image (`testudo.png`) in the `data/` folder.
2. Run the `mbattula_p1.py script:
   ```bash
   python mbattula_p1.py
   ```
3. View the generated outputs in the `outputs/` folder:
   - **`superimposed.avi`**: Video with Testudo image overlaid on the AR tag.
   - **`cube_projection.avi`**: Video showing the 3D cube projected on the AR tag.

---

## **Key Results**

### **1. Tag Detection**
- Accurate detection of AR tags using FFT and Harris Corner Detection.

### **2. Superimposition**
- Seamless replacement of the AR tag with the Testudo image.

### **3. 3D Cube Projection**
- Realistic rendering of a 3D cube on the tag, aligned with the video’s perspective.

---

## **Acknowledgments**
This project was developed as part of the **ENPM673: Perception for Autonomous Robots** course. Special thanks to the instructors and teaching assistants for their guidance.

--- 
