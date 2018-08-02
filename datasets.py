import os

import numpy as np
import torch
import torch.utils.data
from pycocotools.coco import COCO
from PIL import Image


def get_filtered_img_ids(coco, camera_name):
    # use only images that were taken with given camera_name
    cameras = {camera['name']: camera for camera in coco.dataset['cameras']}
    camera_id = cameras[camera_name]['id']
    print('using camera: {}'.format(camera_name))
    imgs = coco.loadImgs(coco.getImgIds())
    def camera_filter(camera_id):
        return lambda x: x['camera_id'] == camera_id
    return [img['id'] for img in filter(camera_filter(camera_id), imgs)]


class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, ann_file, camera_name, object_names, transform):
        self.data_root = data_root
        self.coco = COCO(os.path.join(self.data_root, 'annotations', ann_file))

        img_ids = get_filtered_img_ids(self.coco, camera_name)
        self.object_instances = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_ids))

        self.object_names_map = {cat['id']: cat['name'] for cat in self.coco.dataset['categories']}
        self.object_indices_map = {object_name: i for i, object_name in enumerate(object_names)}
        self.object_ids_map = {cat['name']: cat['id'] for cat in self.coco.dataset['categories']}

        self.transform = transform
    
    def __getitem__(self, index):
        ann = self.object_instances[index]
        img_id = ann['image_id']
        image_file_name = self.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.data_root, 'image', image_file_name))
        image = self.transform(image)
        
        position = [ann['pose']['position']['x'], ann['pose']['position']['y'], ann['pose']['position']['z']]
        orientation = [ann['pose']['orientation']['w'], ann['pose']['orientation']['x'],
                       ann['pose']['orientation']['y'], ann['pose']['orientation']['z']]

        # enforce quaternion [w, x, y, z] to have positive w
        target_pose = np.array(position + orientation, dtype=np.float32)
        if target_pose[3] < 0:
            target_pose[3:] = -target_pose[3:]
        
        object_name = self.object_names_map[ann['category_id']]
        object_index = self.object_indices_map[object_name]
        object_id = self.object_ids_map[object_name]

        return image, target_pose, object_index, object_id

    def __len__(self):
        return len(self.object_instances)


class VisualizationDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, ann_file, camera_name, transform):
        self.data_root = data_root
        self.coco = COCO(os.path.join(self.data_root, 'annotations', ann_file))
        self.img_ids = get_filtered_img_ids(self.coco, camera_name)
        self.transform = transform
    
    def __getitem__(self, index):
        img_id = self.img_ids[index]
        image_file_name = self.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.data_root, 'image', image_file_name))
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_ids)
