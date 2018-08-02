import os
import shutil
import subprocess

import numpy as np
from PIL import Image


def render_wrapper(blender, model_path, pose_list_path, subset_start_index, subset_size, output_dir, camera_parameters, camera_scale, completed_path):
    ret = subprocess.call([blender, '-b', '-P', 'render.py', '--',
                           model_path, pose_list_path,
                           str(subset_start_index), str(subset_size),
                           output_dir,
                           str(camera_parameters['width']), str(camera_parameters['height']),
                           str(camera_parameters['f_x']), str(camera_parameters['f_y']),
                           str(camera_parameters['p_x']), str(camera_parameters['p_y']),
                           str(camera_scale)])
    assert ret == 0
    with open(completed_path, 'a'):
        os.utime(completed_path)


def clean_wrapper(blender, model_path, pose_list_path, subset_start_index, subset_size, output_dir, camera_parameters, camera_scale, completed_path):
    try:
        assert os.path.exists(completed_path)
        assert len(os.listdir(output_dir)) == 2 * subset_size
        for i in range(subset_start_index, subset_start_index + subset_size):
            object_ = Image.open(os.path.join(output_dir, 'object_{:08}.png'.format(i + 1)))
            assert object_.size == (320, 240)
            assert object_.mode == 'RGB'

            mask = Image.open(os.path.join(output_dir, 'mask_{:08}.png'.format(i + 1)))
            assert mask.size == (320, 240)
            assert mask.mode == 'L'
    except:
        try:
            os.remove(completed_path)
        except OSError:
            pass
        try:
            shutil.rmtree(output_dir)
        except OSError:
            pass


def postprocess_wrapper(pose_list_path, completed_dir, image_dir, output_base_dir, camera_name, cat_name, cache_subset_size, train_subset_size, val_subset_size, process_val, num_subsets):
    threshold = 20
    accepted = 0
    total = 0
    poses = np.loadtxt(pose_list_path)

    for mode in ['object', 'mask']:
        poses_dir = os.path.join(output_base_dir, '{}_{}'.format(camera_name, mode), cat_name, 'poses')
        if not os.path.exists(poses_dir):
            os.makedirs(poses_dir)

    def accept_example(cache_subset_dir, index, num_completed_subsets):
        for mode in ['object', 'mask']:
            subset_dir = os.path.join(output_base_dir, '{}_{}'.format(camera_name, mode), cat_name, 'subset_{:08}'.format(num_completed_subsets + 1))
            if not os.path.exists(subset_dir):
                os.makedirs(subset_dir)

            src = os.path.join(cache_subset_dir, '{}_{:08}.png'.format(mode, index + 1))
            dst = os.path.join(subset_dir, os.path.basename(src))
            shutil.move(src, dst)

    num_completed_subsets = 0
    curr_subset = 0
    curr_poses = []
    subset_size = train_subset_size
    i = 0
    # num_subset train subsets plus one more smaller subset for the val set
    while num_completed_subsets < num_subsets + 1:
        completed_path = os.path.join(completed_dir, 'subset_{:08}.txt'.format(i + 1))
        if os.path.exists(completed_path):
            cache_subset_dir = os.path.join(image_dir, 'subset_{:08}'.format(i + 1))
            assert len(os.listdir(cache_subset_dir)) == 2 * cache_subset_size
            for j in range(cache_subset_size):
                index = cache_subset_size * i + j
                mask = Image.open(os.path.join(cache_subset_dir, 'mask_{:08}.png'.format(index + 1)))
                sum_ = np.asarray(mask).sum() / 255
                total += 1
                if sum_ > threshold:
                    accept_example(cache_subset_dir, index, num_completed_subsets)
                    curr_poses.append(poses[index])
                    curr_subset += 1
                    accepted += 1
                    if curr_subset == subset_size:
                        # completed a subset
                        for mode in ['object', 'mask']:
                            poses_dir = os.path.join(output_base_dir, '{}_{}'.format(camera_name, mode), cat_name, 'poses')
                            poses_path = os.path.join(poses_dir, 'subset_{:08}.txt'.format(num_completed_subsets + 1))
                            np.savetxt(poses_path, np.array(curr_poses))
                        num_completed_subsets += 1
                        curr_subset = 0
                        curr_poses = []
                        print('[{}] completed subset {} of {}, total accepted examples: {}/{} ({:.2f}%)'.format(
                            cat_name, num_completed_subsets, num_subsets, accepted, total, 100.0 * accepted / total))
                        if not process_val and num_completed_subsets == num_subsets:
                            return
                        if num_completed_subsets == num_subsets:
                            # switch to val
                            subset_size = val_subset_size
                        if num_completed_subsets == num_subsets + 1:
                            # we are done
                            return
            i += 1
