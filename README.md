## This is the description of the project files: 

1) **main.py**: that file we run by putting only the path of the video.

2) **AllinOne.py**: that file contain the sequence of code functions that run sequentially till the output video produced, it consist of function that run in the main file.

3) **getFrames.py**: first function to run in the AllinOne file that get first 30 secs of the input video.

4) **Optical_Flow/demo.py**: second function to run to get the optical flow of the first 30 secs of the video frames that outputed from getFrames file.

5) **seg.py**: third function to run(it can run parallelaly with demo file) it get the semantic segmentation of the first 30 secs of the video frames that outputed from getFrames file.

6) **predication.py**: fourth function to run that take the output of the demo file, seg file, and getFrame file to predict the driver's focus of attention in the frame but in npz extension it only contain where the focus of attention.

7) **visualiztion.py**: fifth function to run that take the npz file predicted from predication file and put it on the video frames that get from getFrames file and produce jpg images.

8) **frameTovideo.py**: it takes the output jpg images of the visualization file and convert it to a video.

## This is the description of the project folders:

1) **Optical_Flow_Code**: this folder contain the optical flow model code.

2) **light-weight-refinenet-master**: this folder contain the semantic segmentation model code.

3) **model**: this folder contain the predicating driver's focus of attentions model code.

## Additions:

**Youtuve Channel**: [Driver's Focus Of Attention Prediction Videos](https://www.youtube.com/playlist?list=PLpcHTKo4wBZyKHJRIjkMYSKkUdhktd0nz).

**Project Document**: [Predicting Driver's Focus Of Attention Thesis](https://github.com/ahmedtarek97/dreyeve/blob/master/Predicting%20Driver's%20Focus%20Of%20Attention%20Thesis.pdf).

**Demo**: [Demo video of Project](https://youtu.be/TqzLSuzY5d0).

**Model Weights**: [Weights of each branch and tune](https://drive.google.com/drive/folders/1hps4Qm2WwTBXHyMh7KlD_hZpX6hwjvHS?usp=sharing).
