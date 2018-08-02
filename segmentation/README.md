# Segmentation with Dilated Residual Networks (DRNs)

Note: This code is heavily based off of the code release for Dilated Residual Networks located [here](https://github.com/fyu/drn).

## Prerequisites

Please make sure you have downloaded the Oil Change dataset. See the [main README](../README.md) for download instructions.

## Usage

### Pretrained Models

You can download our pretrained segmentation models using the following command:

```bash
./download_pretrained.sh
```

Once they are downloaded, the [eval.ipynb](eval.ipynb) and [visualize.ipynb](visualize.ipynb) notebooks are already configured to evaluate and visualize the pretrained models on our Oil Change dataset.

### Training

The training scripts use Munch config files for starting and resuming training runs. You can reference the [`config`](config) directory for sample config files.

For example, you can start a new training job using one of the config files like so:

```bash
python train.py config/drn_d_22_OilChange.yml
```

Each new training job gets its own log directory and checkpoint directory, time-stamped with the job's start time. By default, log directories are created inside `logs`, and checkpoint directories are created inside `checkpoints`. The training script will write an updated config file to the log directory every time a checkpoint is written. The updated config file will contain all parameters required to resume the run from the last checkpoint.

To resume a training job, simply pass in the corresponding config file:

```bash
python train.py logs/log_dir/config.yml
```

You can modify the `resume` parameter in the config file to resume from a specific checkpoint. Note that no config file will be written to the run's log directory until after the first model checkpoint has been written.

### Monitoring

To monitor training, start TensorBoard and point it at the root log directory:

```bash
tensorboard --logdir logs/
```

You can then view the TensorBoard interface in your browser at `localhost:6006`.

### Evaluating

Please see the [eval.ipynb](eval.ipynb) notebook for code to evaluate a trained model.

### Visualizing

Please see the [visualize.ipynb](visualize.ipynb) notebook to look at example outputs from a trained model.
