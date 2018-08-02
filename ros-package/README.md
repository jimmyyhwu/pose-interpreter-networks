# ROS Package Setup

## Requirements

* [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu)
* [catkin workspace](http://wiki.ros.org/catkin/Tutorials/create_a_workspace)

## Setup

Simply copy the `pose_interpreter_networks` ROS package from this directory to your catkin workspace:

```bash
cp -Lr pose_interpreter_networks/ ~/catkin_ws/src/
```

The ROS package contains many symbolic links to files elsewhere in this repository, so the `-L` flag is required in order to resolve the symlinks when copying.

Note that the requirements for the ROS package are different than those used in the rest of this repository. In particular, ROS does not support Python 3, so you will need to install Python 2 along with the corresponding `pip` dependencies. See the [README](pose_interpreter_networks/README.md) inside the package for more details.

