import os

import numpy as np
import torch
import torch.utils.data
from pycocotools.coco import COCO
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_root, ann_file, split, transforms):
        self.data_root = data_root
        self.split = split
        self.transforms = transforms
        self.coco = COCO(os.path.join(data_root, 'annotations', ann_file))
        self.img_ids = self.coco.getImgIds()
        self.ann_file = ann_file
        self.num_cats = len(self.coco.getCatIds())

    def __getitem__(self, index):
        img = self.coco.loadImgs(self.img_ids[index])[0]
        image = Image.open(os.path.join(self.data_root, 'image', img['file_name']))
        height, width = img['height'], img['width']
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=self.img_ids[index]))
        label = np.zeros((height, width), dtype=np.uint8)
        for ann in anns:
            mask = self.coco.annToMask(ann)
            label[mask == 1] = ann['category_id']
        label = Image.fromarray(label)

        # process kinect_v2 images to be 480x640
        if image.size == (1920, 1080):
            image = image.crop((240, 0, 240 + 1440, 1080))
            label = label.crop((240, 0, 240 + 1440, 1080))
            image = image.resize((640, 480), resample=Image.BILINEAR)
            label = label.resize((640, 480), resample=Image.NEAREST)

        if self.split == 'train':
            image = np.asarray(image)
            image = image / 255.0
            label = np.expand_dims(np.asarray(label), 2)
            image, label = self.transforms([image, label])
            label = label.squeeze(0)
            return (image, label)
        else:
            data = list(self.transforms(image, label))
            return tuple(data)

    def __len__(self):
        return len(self.img_ids)
