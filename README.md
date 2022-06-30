# pose-interpreter-networks

This code release accompanies the following paper:

### Real-Time Object Pose Estimation with Pose Interpreter Networks [[arXiv](https://arxiv.org/abs/1808.01099)] [[video](https://youtu.be/9QBw1NCOOR0)]

Jimmy Wu, Bolei Zhou, Rebecca Russell, Vincent Kee, Syler Wagner, Mitchell Hebert, Antonio Torralba, and David M.S. Johnson

*IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2018

**Abstract:** In this work, we introduce pose interpreter networks for 6-DoF object pose estimation. In contrast to other CNN-based approaches to pose estimation that require expensively annotated object pose data, our pose interpreter network is trained entirely on synthetic pose data. We use object masks as an intermediate representation to bridge real and synthetic. We show that when combined with a segmentation model trained on RGB images, our synthetically trained pose interpreter network is able to generalize to real data. Our end-to-end system for object pose estimation runs in real-time (20 Hz) on live RGB data, without using depth information or ICP refinement.

![](doc/arch.png)
:---:

![](doc/tabletop.gif) | ![](doc/engine.gif)
:---: | :---:

## Overview

File or Directory | Purpose
--- | ---
[`segmentation`](segmentation) | Training, evaluating, and visualizing segmentation models on real RGB data
[`pose_estimation`](pose_estimation) | Training, evaluating, and visualizing pose estimation models (pose interpreter networks) on synthetic data
[`ros-package`](ros-package) | ROS package for real-time object pose estimation on live RGB data
[`end_to_end_eval.ipynb`](end_to_end_eval.ipynb) | Evaluation of end-to-end model on real RGB data
[`end_to_end_visualize.ipynb`](end_to_end_visualize.ipynb) | Demonstration of end-to-end model on real RGB data

## Requirements

These are the basic dependencies (tested on Ubuntu 16.04.4 LTS) for training and evaluating models. Note that some components, such as the ROS package, may have additional/conflicting dependencies. Please see the corresponding READMEs for the specific requirements.

### Python
- python 3.6
- [pytorch](https://pytorch.org/) 0.4 and torchvision 0.2 (`conda install pytorch=0.4 torchvision=0.2 -c pytorch`)
- `pyyaml`
- `munch`
- [COCO API](https://github.com/cocodataset/cocoapi)
- `pypcd` (see [here](https://github.com/jimmyyhwu/pose-interpreter-networks/tree/master/pose_estimation#additional-requirements) for instructions)
- `pillow`
- `scikit-image==0.14.2`
- `matplotlib`
- `tqdm`
- `jupyter`
- `tensorboardX`
- `tensorflow` (for running TensorBoard)

### Blender

Please install [Blender](https://www.blender.org/) 2.79. There is no need to build from source, you can simply download the prebuilt binary and link it at `/usr/local/bin/blender`.

## Getting Started

### Dataset

Download the Oil Change dataset (about 15GB) using the following script:

```bash
./download_data.sh
```

### Pretrained Models

We provide our pretrained models for segmentation and pose estimation. The segmentation model is trained on real RGB data while the pose estimation model (pose interpreter network) is trained entirely on synthetic data. The segmentation and pose estimation models are separately trained and combined into an end-to-end model for pose estimation on real RGB images.

The pretrained segmentation models can be downloaded using these commands:

```bash
cd segmentation
./download_pretrained.sh
```

The pretrained pose estimation models can be similarly downloaded:

```bash
cd pose_estimation
./download_pretrained.sh
```

### Demonstrations

After downloading the pretrained models, please see the following notebooks for demonstrations. In order to run these notebooks, you will need to have all of the requirements listed above installed, including Blender.

Notebook | Purpose
--- | ---
[end_to_end_visualize.ipynb](end_to_end_visualize.ipynb) | Visualize end-to-end model on real RGB data
[segmentation/visualize.ipynb](segmentation/visualize.ipynb) | Visualize pretrained segmentation model on real RGB data
[pose_estimation/demo.ipynb](pose_estimation/demo.ipynb) | Visualize pretrained pose estimation model (pose interpreter network) on synthetic data that is randomly generated on the fly

## Training the System for a New Environment

Our system and pretrained models are tailored towards our own lab testing environment, set of physical objects, and specific Kinect1 camera. To train the system for a new environment, please follow these steps:

* Create a RGB segmentation dataset for your environment
* Train a segmentation model on the RGB dataset
* Calibrate your camera to get the intrinsics
* Create .stl and .pcd 3D mesh models for the objects in your dataset
* Use the camera intrinsics and .stl 3D mesh models to render a synthetic mask image dataset
* Use the synthetic dataset and the .pcd mesh models to train an object mask pose interpreter network
* Combine the segmentation model and pose interpreter network into an end-to-end model

Apart from creation of the RGB segmentation dataset, camera calibration, and 3D model creation, you should be able to adapt the existing code to do all of the other steps.

## Citation

If you find this work useful for your research, please consider citing:

```
@inproceedings{wu2018pose,
  title = {Real-Time Object Pose Estimation with Pose Interpreter Networks},
  author = {Wu, Jimmy and Zhou, Bolei and Russell, Rebecca and Kee, Vincent and Wagner, Syler and Hebert, Mitchell and Torralba, Antonio and Johnson, David M.S.},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year = {2018}
}
```
