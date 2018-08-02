# Synthetic Pose Dataset Rendering

## Prerequisites

Please make sure you have downloaded the Oil Change dataset. See the [main README](../../README.md) for download instructions.

## Additional Requirements

### Python
- python 3.6
- `rq`
- `rq-dashboard`

### Redis

Please visit the official [download page](https://redis.io/download) to install `redis-server`.

## Usage

### Generating Pose Lists

We use a random seed to pregenerate lists of random poses in a deterministic manner. These lists are cached as .txt files. Use the following command to generate pose lists:

```bash
python generate_pose_lists.py
```

### Rendering Images

We use a Redis server and [RQ](http://python-rq.org/) to distribute rendering jobs across many worker processes.

First start up a Redis server:

```bash
redis-server
```

Make a note of the server hostname. For following commands, you can use the `--redis_host` flag indicate the hostname of your Redis server.

Start up some workers:

```bash
python dist_workers.py
```

Distribute the rendering jobs to workers:

```bash
python dist_render.py
```

If any rendering job gets interrupted, we recommend cleaning up corrupt files before restarting the rendering:

```bash
python dist_render.py --clean
```

### Postprocessing Images

In postprocessing, we filter out images where the object is not visible. We start up five postprocessing workers, each handling a single object class.

Start Redis server:

```bash
redis-server
```

Start postprocessing workers:

```bash
python dist_workers.py --queue_name postprocess_queue --num_workers 5
```

Distribute the postprocessing jobs to workers:

```bash
python dist_postprocess.py
```

## Notes

If you would like to evaluate on the same val set we used, please ignore the generated val set and use our provided val set for consistency. This code is set up to generate 320k images, whereas our original code generated 3.2 million images, so a different set of random poses was used to render our val set.
