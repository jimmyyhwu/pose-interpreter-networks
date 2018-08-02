# ROS Package for Real-Time Object Pose Estimation

This ROS package is used to run our pretrained end-to-end model for real-time object pose estimation on live RGB data.

## Requirements

### Python

- python 2.7 (not compatible with Python 3)
- [pytorch](https://pytorch.org/) 0.4 and torchvision 0.2 (`conda install pytorch=0.4 torchvision=0.2 -c pytorch`)
- opencv 3.3 (`conda install opencv`)
- `rospkg`
- `catkin_pkg`
- `pyyaml`
- `munch`
- `matplotlib`

### ROS

Please install the `freenect` ROS package to interface with Kinect1:

```bash
sudo apt-get install ros-kinetic-freenect-stack
```

## Usage

### Configuration

Inside [`param/config.yml`](param/config.yml), you will need to change the following values to reflect the correct paths for your environment:
* `stl_root`
* `segmentation_checkpoint`
* `pose_estimation_checkpoint`

### Launch Files

For all functionalities, the camera driver needs to be started in order to read data from the Kinect1:

```bash
roslaunch pose_interpreter_networks floating_kinect1.launch
```

Additionally, we provide several launch files to run various combinations of functionalities.

To start the live pose estimator only (no segmentation visualization):

```bash
roslaunch pose_interpreter_networks pose_estimator.launch
```

To start the live segmenter only (no pose estimation):

```bash
roslaunch pose_interpreter_networks segmenter.launch
```

To start the live pose estimator with segmentation visualization (runs slower due to visualization overhead):

```bash
roslaunch pose_interpreter_networks pose_estimator_pub_segmentation.launch
```

### Visualization in RViz

Please start RViz to visualize the input image feed from the camera and the output segmentations or pose estimates:

```bash
rviz
```

We have provided our rviz config file in the `rviz` directory, which you can copy into your own rviz config directory:

```bash
mkdir -p ~/.rviz/
cp rviz/default.rviz ~/.rviz/
```

Alternatively, if you are starting with the default rviz configuration, you can make the following changes to the interface yourself:

1. In the Display panel, under Global Options, set Fixed Frame to `floating_kinect1_rgb_optical_frame`
2. Add a Camera display for topic `/floating_kinect1/rgb/image_rect_color`
3. Add an Image display for topic `/output/segmentation` to view the segmentation output
4. Add a MarkerArray display for topic `/output/pose_estimate` to visualize the pose estimates. RViz should transform the 3D object meshes according to the pose estimates and overlay them onto the Camera display.
5. The MarkerArray overlay may be too dark. To fix this, go to the Views panel and change the Orbit yaw and pitch to both be -1.57.
