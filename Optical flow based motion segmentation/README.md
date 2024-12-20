# **Motion Segmentation Using Optical Flow**
## **Introduction**:
This project focuses on leveraging RAFT (Recurrent All-Pairs Field Transforms), a state-of-the-art optical flow algorithm, to perform motion segmentation. Optical flow is utilized to differentiate between static and moving objects within video sequences. Our motion segmentation approach is inspired by the probabilistic model proposed in the paper "It’s Moving!" by Bideau et al. The KITTI MOTS (Multi-Object Tracking and Segmentation) dataset provides ground truth for evaluating segmentation accuracy, ensuring robust validation.

## **Requirements**

To run the project, the following modules and software are required:

### **Python Dependencies**:
- **pytorch**
- **opencv**
- **scipy**

You can install the required dependencies using conda:

```bash
conda install pytorch=1.6.0 torchvision=0.7.0 matplotlib tensorboard scipy opencv -c pytorch
```
## **Requirements**
Raft requires the following modules.

- pytorch
- opencv
- scipy

To install you can do the following with conda

```Shell
conda install pytorch=1.6.0 torchvision=0.7.0 matplotlib tensorboard scipy opencv -c pytorch
```
- Matlab is required to do motion-segmentation from optical flow


# **Project Overview**
### Optical Flow with RAFT:
RAFT is a cutting-edge technique in computer vision that computes pixel-level optical flow between consecutive frames. Its key strengths include:

- State-of-the-art accuracy for optical flow estimation.
- Generalization across datasets, enabling reliable results in diverse conditions.
- High efficiency, ideal for motion segmentation applications.
### **The algorithm comprises three main components:**

- Feature Encoder: Extracts pixel-level features from video frames.
- Correlation Layer: Produces multi-scale 4D correlation volumes for all pixel pairs.
- Recurrent GRU Unit: Iteratively updates the flow field using correlation values.
## Motion Segmentation Pipeline
### Step 1: Data Preparation
The dataset is organized as follows:

```Shell
├── data
    ├── vedio1
        ├── frames
        ├── optical_flow
        ├── vedio.mp4
    ├── vedio2
        ├── frames
        ├── optical_flow
    .
    .
    .
```

Short-duration videos are preferred to optimize computational resources.

### Step 2: Optical Flow Estimation
Generate optical flow using the RAFT algorithm:

```Shell
python3 optical_flow.py vedio1
```

This command computes and saves frame-by-frame optical flow outputs in the optical_flow directory.

### Step 3: Motion Segmentation
MATLAB Execution:
Open the segment.m file.
Specify the video name in the runSegmentation function. Example

```shell
runSegmentation('video1')
```

Execute the script in MATLAB with the dataset folder as the working directory.
### Output:
Results are saved in the results folder.
# Technical Highlights:
### 1. Segmentation Criteria:

Pixels are labeled as either static or moving objects.
Entire objects are segmented even if only parts exhibit motion (e.g., a walking person with a stationary foot).

### 2. Background Motion Estimation:

RANSAC-based initialization estimates translation and rotation parameters for background motion.
Superpixels are used to enhance robustness in corner regions, minimizing errors due to camera rotation.

### 3. Flow Angle Likelihood:

Motion segmentation relies on the flow angle rather than magnitude for accuracy, as angles contain the most reliable directional information.
A probabilistic model adjusts segmentation based on angle likelihood conditioned by flow magnitude.

### 4. Posterior Propagation:

Motion priors are propagated from previous frames using Gaussian smoothing and renormalization, allowing the model to dynamically adapt to scene changes.

## Challenges and Results:

### Challenges:
- Low Video Quality: Data collected using a Pi Camera yielded suboptimal optical flow results, affecting segmentation quality.
- Computation Time: The segmentation process, from optical flow computation to MATLAB-based motion segmentation, is time-intensive.
### Results:
Despite these challenges, our approach demonstrated effective segmentation for short-duration videos. The outputs showcase successful differentiation between static and dynamic objects. Results can be viewed in the results folder or the accompanying video files:

### Video Results:
<p align="center">
  <img src="Optical flow based motion segmentation\results\video1\video1.gif" width="50%">
</p>

<p align="center">
  <img src="Optical flow based motion segmentation\results\video2\traffic.gif" width="50%">
</p>

<p align="center">
  <img src="Optical flow based motion segmentation\results\video3\video3.gif" width="50%">
</p>
---


