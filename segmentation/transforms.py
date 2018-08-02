# Authors: David M.S. Johnson, Abraham Schneider, Rebecca Russell
#
# Partially adapted from https://github.com/fyu/drn/blob/master/data_transforms.py

from __future__ import division

import numpy as np
import torch
from skimage.exposure import adjust_gamma
from skimage.transform import resize, rotate


class CropTransform(object):
    def __init__(self, shape=None, cropGenerator=None, paddingGenerators=None):
        if cropGenerator:
            self.cropGenerator = cropGenerator
            self.paddingGenerators = paddingGenerators
        else:
            self.cropGenerator = RandomCropGenerator(*shape)
            self.paddingGenerators = [RandomFloatGenerator(), RandomFloatGenerator()]

    @property
    def generators(self):
        return [self.cropGenerator] + self.paddingGenerators

    def __call__(self, data):
        x, y = self.cropGenerator.value
        width = self.cropGenerator.width
        height = self.cropGenerator.height

        if data.shape[1] < width:
            padding_left_frac = self.paddingGenerators[0].value
            diff = (width - data.shape[1])
            padding_left = int(diff*padding_left_frac)
            padding_right = diff - padding_left
        else:
            padding_left = 0
            padding_right = 0

        if data.shape[0] < height:
            padding_top_frac = self.paddingGenerators[1].value
            diff = height - data.shape[0]
            padding_top = int(diff*padding_top_frac)
            padding_bottom = diff - padding_top
        else:
            padding_top = 0
            padding_bottom = 0

        data = np.pad(data, ((padding_top, padding_bottom), (padding_left, padding_right), (0, 0)), "constant")
        return data[y:(y+height), x:(x+width), :]


class RandomCropGenerator(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, data):
        width = data.shape[1]
        height = data.shape[0]

        if self.width < width:
            self.x = np.random.randint(0, width - self.width)
        else:
            self.x = 0

        if self.height < height:
            self.y = np.random.randint(0, height - self.height)
        else:
            self.y = 0

    @property
    def value(self):
        return (self.x, self.y)


class RandomFloatGenerator(object):
    def __call__(self, data):
        self.value = torch.rand(1)[0]


class RandomIntGenerator(object):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, data):
        self.value = np.random.randint(self.min, self.max)


class ZoomTransform(object):
    def __init__(self, interpolation="bilinear", zoom=None, generator=None):
        if interpolation == "nearest":
            self.interp_order = 0
        elif interpolation == "bilinear":
            self.interp_order = 1

        if generator:
            self.zoomGenerator = generator
        else:
            self.zoomGenerator = RandomIntGenerator(*zoom)

    @property
    def generators(self):
        return [self.zoomGenerator]

    def __call__(self, data):
        width = self.zoomGenerator.value
        ratio = data.shape[1] / data.shape[0]
        height = int(width*ratio)
        result = resize(data, (width, height, data.shape[2]), order=self.interp_order, mode="constant", preserve_range=True)
        return result


class FrequencyTransform(object):
    def __init__(self, freq, transform, generator=None):
        self.transform = transform
        self.freq = freq

        if generator:
            self.generator = generator
        else:
            self.generator = RandomFloatGenerator()

    @property
    def generators(self):
        return [self.generator] + self.transform.generators

    def __call__(self, data):
        if self.generator.value < self.freq:
            return self.transform(data)

        return data


class RotateTransform(object):
    def __init__(self, interpolation, generator=None):
        if generator:
            self.generator = generator
        else:
            self.generator = RandomFloatGenerator()

        if interpolation == "nearest":
            self.interp_order = 0
        elif interpolation == "bilinear":
            self.interp_order = 1

    @property
    def generators(self):
        return [self.generator]

    def __call__(self, data):
        return rotate(data, 360*self.generator.value, resize=False, preserve_range=True, order=self.interp_order)


class BrightnessTransform(object):
    def __init__(self, minValue, maxValue, generator=None):
        self.minValue = minValue
        self.maxValue = maxValue

        if generator:
            self.generator = generator
        else:
            self.generator = RandomFloatGenerator()

    @property
    def generators(self):
        return [self.generator]

    def __call__(self, data):
        range_ = self.maxValue - self.minValue
        value = (self.generator.value*range_) + self.minValue
        return adjust_gamma(data, gamma=value)


class ToTensorTransform(object):
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, data):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
            data = data.permute(2, 0, 1)

        if self.dtype != None:
            data = data.type(self.dtype)

        return data


class ParallelTransform(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def _generate(self, transform, data):
        try:
            for gen in transform.generators:
                gen(data)
        except Exception:
            # if no generator exists, ignore
            pass

    def _transform(self, transforms, data, i):
        for transform in transforms:
            if i == 0: self._generate(transform, data)
            if transform != None:
                data = transform(data)
        return data

    def __call__(self, data):
        return [self._transform(self.transforms[i], d, i) for i, d in enumerate(data)]


class ToTensor(object):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic, label=None):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
            if pic.mode == 'YCbCr':

                nchannel = 3
            else:
                nchannel = len(pic.mode)
            img = img.view(pic.size[1], pic.size[0], nchannel)
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255)
        if label is None:
            return img,
        else:
            return img, torch.LongTensor(np.array(label, dtype=np.int))


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
