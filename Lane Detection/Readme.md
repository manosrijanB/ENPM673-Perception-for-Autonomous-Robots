# ****Lane Detection****

## **Introduction**

This project consists of three main tasks: **Histogram Equalization**, **Straight Lane Detection**, and **Turn Prediction**. Each task focuses on improving visual data quality, detecting lane types, and predicting road curvature to emulate components of self-driving car systems. Below is an overview of the project implementation and its outcomes.

---

## **Project Components**

### **1. Histogram Equalization**
**Objective**: Enhance the contrast and visual appearance of a video sequence captured under low-light conditions, which is essential for downstream tasks like lane detection.

**Approach**:
- Convert the video frames to **LAB color space**.
- Apply:
  - **Histogram Equalization**: Distributes intensities evenly across the image.
  - **Adaptive Histogram Equalization (AHE)**: Divides the image into smaller grids and applies local histogram equalization for better contrast improvement in uneven lighting.
- Merge the processed 'L' channel back with 'A' and 'B'.

**Results**:
- Normal Histogram Equalization improved contrast significantly.
- AHE further highlighted details in darker regions, revealing more information about objects like trees, vehicles, and pedestrians.

<p align="center">
  <img src="Lane Detection\output_videos\adaptiveequilized.gif" width="50%">
</p>
---

### **2. Straight Lane Detection**
**Objective**: Detect and classify straight lanes in a video sequence into solid (green) and dashed (red) lines, mimicking a Lane Departure Warning System.

**Approach**:
1. **Preprocessing**:
   - Convert frames to grayscale.
   - Apply **Gaussian Blur** to reduce noise.
   - Detect edges using **Canny Edge Detection**.
2. **Region of Interest (ROI)**: Mask irrelevant regions to focus on lanes.
3. **Lane Detection**:
   - Use the **HoughLines** algorithm to detect lane lines.
   - Differentiate between left and right lanes based on slope:
     - Negative slope: Left lane.
     - Positive slope: Right lane.
4. **Line Classification**:
   - Solid lines: Green.
   - Dashed lines: Red.

**Results**:
- Successfully identified and classified lanes in the video sequence.
- The algorithm can generalize to horizontally flipped videos due to its reliance on slope and position.

<p align="center">
  <img src="Lane Detection\output_videos\Lane Detection.gif" width="50%">
</p>
---

### **3. Predict Turn**
**Objective**: Detect curved lanes, predict turns (left or right), and compute the radius of curvature.

**Approach**:
1. **Preprocessing**:
   - Filter frames to highlight yellow and white lane markings.
   - Convert to grayscale and mask the region of interest.
2. **Perspective Transformation**:
   - Apply **Warp Perspective** to achieve a bird's-eye view.
3. **Lane Detection**:
   - Use **Polyfit** to fit lane lines.
   - Superimpose the detected lanes on the original video.
4. **Turn Prediction**:
   - Compute the radius of curvature using the polynomial coefficients.
   - Predict turn direction based on curvature sign:
     - Positive: Right turn.
     - Negative: Left turn.
5. **Error Handling**:
   - Use the previous frames' history to extrapolate missing lane lines when detection fails.

**Results**:
- The system accurately predicted turns and visualized lane curvature.
- Successfully handled edge cases, such as temporary occlusion or lane loss.

<p align="center">
  <img src="Lane Detection\output_videos\predict_turn.gif" width="50%">
</p>
---

## **Challenges and Observations**

1. **Histogram Equalization**:
   - AHE was computationally expensive but provided significantly better results for uneven lighting.
2. **Lane Detection**:
   - The slope-based approach was effective but sensitive to noise in edge detection.
3. **Turn Prediction**:
   - Perspective transformation was critical for accurate curvature calculation.
   - Handling abrupt lane changes required smoothing across frames.

---

## **Directory Structure**

```bash
├── data
    ├── video1
        ├── frames
        ├── optical_flow
        ├── video.mp4
    ├── video2
        ├── frames
        ├── optical_flow
├── scripts
    ├── histogram_equalization.py
    ├── lane_detection.py
    ├── turn_prediction.py
├── results
    ├── video1_results.mp4
    ├── video2_results.mp4
    ├── hist_equalization_comparison.png
    ├── turn_prediction_overlay.mp4
├── README.md
├── report.pdf
```

---

## **How to Run**

### **1. Preprocess Video**
Use the provided scripts for each task:
- **Histogram Equalization**:
  ```bash
  python histogram_equalization.py --input data/video1/video.mp4 --output results/video1_histogram.mp4
  ```

- **Lane Detection**:
  ```bash
  python lane_detection.py --input data/video1/video.mp4 --output results/video1_lanes.mp4
  ```

- **Turn Prediction**:
  ```bash
  python turn_prediction.py --input data/video1/video.mp4 --output results/video1_turns.mp4
  ```

### **2. View Results**
All results are saved in the `results` folder. Videos and images demonstrate the enhanced quality, lane detection, and turn prediction outcomes.

---

## **Acknowledgments**
- Datasets: KITTI MOTS, Udacity Advanced Lane Detection dataset.
- Tools: OpenCV, NumPy, Python.

For more details, refer to the [project report](report.pdf) or view the sample outputs in the `results` folder.