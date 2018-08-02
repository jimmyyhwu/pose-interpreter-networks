import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


def accuracy(output, target):
    _, pred = output.max(1)
    pred = pred.view(-1)
    target = target.view(-1)
    correct = pred.eq(target)
    score = correct.float().sum(0).mul(100.0 / correct.size(0))
    return score.item()


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iou(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


object_colors = [
    'k',  # 0 background
    'm',  # 1 oil_bottle
    'w',  # 2 fluid_bottle
    'c',  # 3 oilfilter
    'g',  # 4 funnel
    'b',  # 5 engine
    'r',  # 6 blue_funnel
    'orange',  # 7 tissue_box
    'brown',  # 8 drill
    'lime',  # 9 cracker_box
    'yellow'   # 10 spam
]
cmap = colors.ListedColormap(object_colors)


def visualize(ax, image, label):
    ax.imshow(image)
    ax.imshow(label, cmap=cmap, alpha=0.5, vmin=0, vmax=len(object_colors) - 1)


def render_batch(visualize_fn, input, target, output):
    batch_size = input.shape[0]
    fig, axes = plt.subplots(nrows=batch_size, ncols=3, figsize=(12, 4*batch_size))
    plt.subplots_adjust(left=0.05, bottom=0, right=0.95, top=1, hspace=0)
    for i in range(batch_size):
        ax = axes if batch_size == 1 else axes[i]  # otherwise won't work if nrows is 1
        ax[0].imshow(input[i])
        visualize_fn(ax[1], input[i], target[i])
        visualize_fn(ax[2], input[i], output[i])
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='') / 255.
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data


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
