# Motion Segmentation based on Optical Flow

## Team Members
- Bharath Chandra (118482804)
- Saketh Bangiri (118548814)
- Balaji Selvakumar(118545745)
- Mano Srijan (118546490)

## Requirements
Raft requires the following modules.

- pytorch
- opencv
- scipy

To install you can do the following with conda

```Shell
conda install pytorch=1.6.0 torchvision=0.7.0 matplotlib tensorboard scipy opencv -c pytorch
```
- Matlab is required to do motion-segmentation from optical flow

## To Do Motion Segmentation:

Dataset folder has the following structure.

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
The whole segmentation process takes so much time, so short videos are used.

To generate optical flow 
```Shell
python3 optical_flow.py vedio1
```
 - Now optical flow is displayed and saved frame by frame in the optical_flow folder in the dataset

For Motion Segmentaion
- open segment.m file
- edit the input to runSegmentation to the video used above.
- example: runSegmentation('vedio1')
- Run this file matlab as current folder as working directory.
- Result will be saved in results folder.
