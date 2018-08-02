# Pose Estimation with Pose Interpreter Networks

## Prerequisites

Please make sure you have downloaded the Oil Change dataset. See the [main README](../README.md) for download instructions.

## Additional Requirements

### Python

Please install the `pypcd` package. Pull request [#9](https://github.com/dimatura/pypcd/pull/9) needs to be patched in to enable Python 3 compatibility.

The full installation commands are as follows:

```bash
git clone https://github.com/dimatura/pypcd.git
cd pypcd
git fetch origin pull/9/head:python3-fix
git checkout python3-fix
python setup.py install
```

### Point Cloud Library (Optional)

This training code uses .pcd files to represent object point clouds, which are used in the proposed point cloud loss function. If you would like to generate your own .pcd files, you will need to follow the tutorial [here](http://www.pointclouds.org/documentation/tutorials/compiling_pcl_posix.php) to compile PCL from source. The specific tools of interest are [`pcl_converter`](https://github.com/PointCloudLibrary/pcl/blob/master/io/tools/converter.cpp) and [`pcl_mesh2pcd`](https://github.com/PointCloudLibrary/pcl/blob/master/tools/mesh2pcd.cpp). Alternatively, you can simply download our .pcd files for the objects we used:

```bash
./download_pcds.sh
```

## Usage

### Dataset

Use the following command to download our synthetic image dataset:

```bash
./download_data.sh
```

Note that while our pretrained models were trained on 3.2 million images, we only provide the first 320k images in the dataset, as we found that having additional training images does not further improve model performance. Alternatively, if you would like to generate your own dataset, you can reference the [`dataset`](dataset) directory for further instructions.

### Pretrained Models

You can download our pretrained pose interpreter networks using the following command:

```bash
./download_pretrained.sh
```

Once they are downloaded, the [eval.ipynb](eval.ipynb) and [visualize.ipynb](visualize.ipynb) notebooks are already configured to evaluate and visualize the pretrained models on the synthetic image dataset. Additionally, the [demo.ipynb](demo.ipynb) notebook shows how to visualize the pretrained models on images that are randomly generated on the fly.

### Training

The training script setup is very similar to that of the segmentation training script. See the segmentation [README](../segmentation/README.md) for more details.

To start a new training job:

```bash
python train.py config/floating_kinect1_mask.yml
```

To resume a training job:

```bash
python train.py logs/log_dir/config.yml
```

### Monitoring

To monitor training, start TensorBoard and point it at the root log directory:

```bash
tensorboard --logdir logs/
```

You can then view the TensorBoard interface in your browser at `localhost:6006`.

### Evaluating

Please see the [eval.ipynb](eval.ipynb) notebook for code to evaluate a trained model.

### Visualizing

Please see the [visualize.ipynb](visualize.ipynb) notebook to load a trained model and visualize model outputs on the val set of the synthetic image dataset. Alternatively, you can look at the [demo.ipynb](demo.ipynb) notebook to visualize model outputs on synthetic test images that are randomly generated on the fly.

## Notes

An `epoch` in the pose interpreter network training code doesn't actually refer to an epoch through the synthetic dataset. Instead, it refers to a batch of 32k training images (1000 iterations with batch size 32). Our pretrained models were trained on 3.2 million images, divided into 100 subsets of 32k images each, so a single epoch through the 3.2 million training images is actually 100 `epoch`s in the code. However, we only provide 320k training images for download, which we found was sufficient to attain comparable performance to using the full 3.2 million. Conveniently, even though more actual epochs are required since the dataset is smaller, the number of iterations required is the same, so you will not need to change the `epoch`s required in the code.
