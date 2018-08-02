import os
import subprocess
import tempfile

import numpy as np
import torch
import torch.utils.data
from pycocotools.coco import COCO
from pypcd import pypcd
from PIL import Image


def get_pc(pcd_root, pcd_name, downsample_factor=1):
    pcd_path = os.path.join(pcd_root, pcd_name)
    pcd = pypcd.PointCloud.from_path(pcd_path)
    points = np.ones((pcd.points // downsample_factor, 4), dtype=np.float32)
    points[:, :3] = pcd.pc_data.view((np.float32, 3))[downsample_factor-1::downsample_factor, :]
    return points.T


def batch_rotation_angle(q1, q2):
    return torch.acos(torch.clamp(2 * (q1 * q2).sum(dim=1).pow(2) - 1, max=1))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_camera_parameters(oil_change_data_root, ann_file, camera_name):
    coco = COCO(os.path.join(oil_change_data_root, 'annotations', ann_file))
    cameras = {camera['name']: camera for camera in coco.dataset['cameras']}
    camera_info = cameras[camera_name]
    camera_parameters = {
        'width': camera_info['width'],
        'height': camera_info['height'],
        'f_x': camera_info['K'][0],
        'f_y': camera_info['K'][4],
        'p_x': camera_info['K'][2],
        'p_y': camera_info['K'][5]
    }
    return camera_parameters


def get_model_paths(oil_change_data_root, ann_file, object_names):
    coco = COCO(os.path.join(oil_change_data_root, 'annotations', ann_file))
    model_paths_map = {
        cat['name']: os.path.join(oil_change_data_root, 'meshes', cat['mesh'])
        for cat in coco.dataset['categories']
    }
    return [model_paths_map[object_name] for object_name in object_names]


class PoseRenderer:
    def __init__(self, blender_path, camera_parameters, model_path, mode, camera_scale=0.5):
        self.blender_path = blender_path

        self.camera_parameters = camera_parameters
        self.model_path = model_path
        self.mode = mode
        self.camera_scale = camera_scale

        cwd = os.path.dirname(os.path.realpath(__file__))
        self.script_path = os.path.join(cwd, 'render_pose.py')

    def render(self, position, orientation):
        position = ','.join(map(str, position))
        orientation = ','.join(map(str, orientation))

        with tempfile.TemporaryDirectory() as cache_dir:
            output_path = os.path.join(cache_dir, 'render.png')
            ret = subprocess.call([self.blender_path, '-b', '-P', self.script_path, '--',
                                   self.model_path, output_path, self.mode,
                                   str(self.camera_parameters['width']), str(self.camera_parameters['height']),
                                   str(self.camera_parameters['f_x']), str(self.camera_parameters['f_y']),
                                   str(self.camera_parameters['p_x']), str(self.camera_parameters['p_y']),
                                   str(self.camera_scale),
                                   position, orientation])
            assert ret == 0
            image = np.asarray(Image.open(output_path))

        return image
