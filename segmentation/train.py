import matplotlib
matplotlib.use('Agg')

import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from munch import Munch
from tensorboardX import SummaryWriter
from torch import nn

import datasets
import models
import transforms
import utils

import sys
assert sys.version.startswith('3.6')
assert torch.__version__.startswith('0.4')


def adjust_learning_rate(optimizer, epoch):
    lr = cfg.optimizer.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(checkpoint_dir, state, epoch):
    file_path = os.path.join(checkpoint_dir, 'checkpoint_{:08d}.pth.tar'.format(epoch))
    torch.save(state, file_path)
    return file_path


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        output = model(input)
        loss = criterion(output, target)

        losses.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.training.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses))

    return batch_time.avg, data_time.avg, losses.avg


def validate(val_loader, model, criterion):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    acc = utils.AverageMeter()
    model.eval()
    hist = np.zeros((cfg.data.classes, cfg.data.classes))

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            data_time.update(time.time() - end)

            target = target.cuda(non_blocking=True)

            output = model(input)
            loss = criterion(output, target)

            losses.update(loss.item(), input.size(0))
            acc.update(utils.accuracy(output, target), input.size(0))

            _, pred = output.max(1)
            hist += utils.fast_hist(pred.cpu().data.numpy().flatten(), target.cpu().numpy().flatten(), cfg.data.classes)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg.training.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {accuracy.val:.4f} ({accuracy.avg:.4f})'.format(
                          i, len(val_loader), batch_time=batch_time, data_time=data_time, loss=losses, accuracy=acc))

    ious = utils.per_class_iou(hist) * 100
    return batch_time.avg, data_time.avg, losses.avg, acc.avg, ious


def visualize_batch(visualize_fn, model, input, target):
    with torch.no_grad():
        output = model(input).cpu()
        _, output = output.max(1)
        output = output.data.numpy()

        #for i in xrange(input.size(0)):
        #    for t, m, s in zip(input[i], cfg.data.mean, cfg.data.std):
        #        t.mul_(s).add_(m)
        #input.clamp_(0, 1)
        input = input.permute(0, 2, 3, 1).numpy()
        target = target.numpy()
    return utils.render_batch(visualize_fn, input, target, output)


def main(cfg):
    if cfg.training.resume is not None:
        log_dir = cfg.training.log_dir
        checkpoint_dir = os.path.dirname(cfg.training.resume)
    else:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
        log_dir = os.path.join(cfg.training.logs_dir, '{}_{}'.format(timestamp, cfg.training.experiment_name))
        checkpoint_dir = os.path.join(cfg.training.checkpoints_dir, '{}_{}'.format(timestamp, cfg.training.experiment_name))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print('log_dir: {}'.format(log_dir))
        print('checkpoint_dir: {}'.format(checkpoint_dir))

    single_model = models.DRNSeg(cfg.arch, cfg.data.classes, None, pretrained=True)
    model = torch.nn.DataParallel(single_model).cuda()
    cudnn.benchmark = True
    criterion = nn.NLLLoss().cuda()
    optimizer = torch.optim.SGD(single_model.optim_parameters(),
                                cfg.optimizer.lr,
                                momentum=cfg.optimizer.momentum,
                                weight_decay=cfg.optimizer.weight_decay)
    start_epoch = 0
    if cfg.training.resume is not None:
        if os.path.isfile(cfg.training.resume):
            print("=> loading checkpoint '{}'".format(cfg.training.resume))
            checkpoint = torch.load(cfg.training.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(cfg.training.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(cfg.training.resume))

    crop_transform = transforms.CropTransform(shape=(640, 480))
    zoom_generator = transforms.RandomIntGenerator(480, 540)
    zoom_bilinear_transform = transforms.ZoomTransform(interpolation="bilinear", generator=zoom_generator)
    zoom_nearest_transform = transforms.ZoomTransform(interpolation="nearest", generator=zoom_generator)
    rotate_freq_generator = transforms.RandomFloatGenerator()
    rotate_angle_generator = transforms.RandomFloatGenerator()
    rotate_bilinear_transform = transforms.FrequencyTransform(
        freq=0.5,
        transform=transforms.RotateTransform(interpolation="bilinear", generator=rotate_angle_generator),
        generator=rotate_freq_generator
    )
    rotate_nearest_transform = transforms.FrequencyTransform(
        freq=0.5,
        transform=transforms.RotateTransform(interpolation="nearest", generator=rotate_angle_generator),
        generator=rotate_freq_generator
    )
    brightness_generator = transforms.RandomFloatGenerator()
    gamma_transform = transforms.BrightnessTransform(0.5, 1.5, brightness_generator)
    train_image_transforms = (zoom_bilinear_transform, rotate_bilinear_transform, crop_transform, gamma_transform, transforms.ToTensorTransform(torch.FloatTensor))
    label_transforms = (zoom_nearest_transform, rotate_nearest_transform, crop_transform, transforms.ToTensorTransform(torch.LongTensor))

    train_transforms = transforms.ParallelTransform([train_image_transforms, label_transforms])
    val_transforms = transforms.Compose([transforms.ToTensor()])

    if cfg.data.train_all:
        train_dataset = datasets.Dataset(cfg.data.root, cfg.data.ann_file, 'train', train_transforms)
    else:
        train_dataset = datasets.Dataset(cfg.data.root, 'train_' + cfg.data.ann_file, 'train', train_transforms)
    val_dataset = datasets.Dataset(
        cfg.data.root, 'val_' + cfg.data.ann_file, 'val', val_transforms)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.workers, pin_memory=True)

    train_summary_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    val_summary_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))
    visualization_summary_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'visualization'))
    for epoch in range(start_epoch, cfg.training.epochs):
        lr = adjust_learning_rate(optimizer, epoch)
        train_summary_writer.add_scalar('learning_rate', lr, epoch + 1)

        train_batch_time, train_data_time, train_loss = train(train_loader, model, criterion, optimizer, epoch)
        train_summary_writer.add_scalar('batch_time', train_batch_time, epoch + 1)
        train_summary_writer.add_scalar('data_time', train_data_time, epoch + 1)
        train_summary_writer.add_scalar('loss', train_loss, epoch + 1)

        val_batch_time, val_data_time, val_loss, val_accuracy, val_ious = validate(val_loader, model, criterion)
        val_summary_writer.add_scalar('batch_time', val_batch_time, epoch + 1)
        val_summary_writer.add_scalar('data_time', val_data_time, epoch + 1)
        val_summary_writer.add_scalar('loss', val_loss, epoch + 1)
        val_summary_writer.add_scalar('accuracy', val_accuracy, epoch + 1)
        for i, iou in enumerate(val_ious):
            if not np.isnan(iou) and iou != 0:
                val_summary_writer.add_scalar('iou_{}'.format(cfg.data.class_names[i]), iou, epoch + 1)

        first_input_batch, first_target_batch = iter(val_loader).next()
        rendered = visualize_batch(utils.visualize, model, first_input_batch, first_target_batch)
        visualization_summary_writer.add_image('segmentation', torch.from_numpy(rendered).permute(2, 0, 1), epoch + 1)

        if (epoch + 1) % cfg.training.checkpoint_epochs == 0:
            checkpoint_path = save_checkpoint(checkpoint_dir, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, epoch + 1)
            cfg.training.log_dir = log_dir
            cfg.training.resume = checkpoint_path
            with open(os.path.join(log_dir, 'config.yml'), 'w') as f:
                f.write(cfg.toYAML())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', metavar='PATH', help='path to config file')
    args = parser.parse_args()
    config_path = args.config_path
    with open(config_path, 'r') as f:
        cfg = Munch.fromYAML(f)
    main(cfg)
