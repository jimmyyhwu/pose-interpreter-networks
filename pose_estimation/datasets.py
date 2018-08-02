import os

import numpy as np
import torch
import torch.utils.data
from skimage.draw import circle
from skimage.measure import find_contours
from PIL import Image


class RenderedPoseDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, objects, subset_num, transform):
        self.transform = transform

        # images
        image_dirs = []
        self.object_indices = []

        for o in objects:
            image_dirs.append(os.path.join(data_root, o, 'subset_{:08}'.format(subset_num)))
        for image_dir in image_dirs:
            assert os.path.exists(image_dir)
        self.image_paths = []
        for i, image_dir in enumerate(image_dirs):
            image_names = sorted(os.listdir(image_dir))
            self.image_paths.extend([os.path.join(image_dir, name) for name in image_names])
            self.object_indices.extend(i * np.ones(len(image_names)))
        self.object_indices = np.array(self.object_indices, dtype=np.int64)
        assert len(self.object_indices) == len(self.image_paths)

        # poses
        poses_paths = []
        for o in objects:
            poses_paths.append(os.path.join(data_root, o, 'poses', 'subset_{:08}.txt'.format(subset_num)))
        for poses_path in poses_paths:
            assert os.path.exists(poses_path)
        self.poses = []
        for poses_path in poses_paths:
            self.poses.extend(np.loadtxt(poses_path).astype(np.float32))
        assert len(self.poses) == len(self.image_paths)

    def __getitem__(self, index):
        object_index = self.object_indices[index]

        image = Image.open(self.image_paths[index])
        image = self.transform(image)

        # enforce quaternion [w, x, y, z] to have positive w
        target_pose = self.poses[index]
        if target_pose[3] < 0:
            target_pose[3:] = -target_pose[3:]

        return image, target_pose, object_index

    def __len__(self):
        return len(self.image_paths)


class OccludedRenderedPoseDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, objects, subset_num, transform, max_circle_size):
        self.transform = transform
        self.max_circle_size = max_circle_size

        # images
        image_dirs = []
        self.object_indices = []

        for o in objects:
            image_dirs.append(os.path.join(data_root, o, 'subset_{:08}'.format(subset_num)))
        for image_dir in image_dirs:
            assert os.path.exists(image_dir)
        self.image_paths = []
        for i, image_dir in enumerate(image_dirs):
            image_names = sorted(os.listdir(image_dir))
            self.image_paths.extend([os.path.join(image_dir, name) for name in image_names])
            self.object_indices.extend(i * np.ones(len(image_names)))
        self.object_indices = np.array(self.object_indices, dtype=np.int64)
        assert len(self.object_indices) == len(self.image_paths)

        # poses
        poses_paths = []
        for o in objects:
            poses_paths.append(os.path.join(data_root, o, 'poses', 'subset_{:08}.txt'.format(subset_num)))
        for poses_path in poses_paths:
            assert os.path.exists(poses_path)
        self.poses = []
        for poses_path in poses_paths:
            self.poses.extend(np.loadtxt(poses_path).astype(np.float32))
        assert len(self.poses) == len(self.image_paths)

    def __getitem__(self, index):
        object_index = self.object_indices[index]

        image = Image.open(self.image_paths[index])

        # if possible, occlude the object
        np_image = np.array(image)
        contours = find_contours(np_image.mean(axis=2) if np_image.ndim == 3 else np_image, 0)
        if len(contours) > 0:
            contour = sorted(contours, key=lambda x: -x.shape[0])[0]
            if len(contour) > 0:
                occluded_image = np_image.copy()
                circle_center = contour[np.random.choice(len(contour))]
                r, c = circle_center
                circle_size = np.random.randint(self.max_circle_size + 1)
                rr, cc = circle(r, c, circle_size, shape=np_image.shape)
                occluded_image[rr, cc] = 0
                image = Image.fromarray(occluded_image)

        image = self.transform(image)

        # enforce quaternion [w, x, y, z] to have positive w
        target_pose = self.poses[index]
        if target_pose[3] < 0:
            target_pose[3:] = -target_pose[3:]

        return image, target_pose, object_index

    def __len__(self):
        return len(self.image_paths)
