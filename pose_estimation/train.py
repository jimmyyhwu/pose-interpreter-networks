import argparse
import os
import time
from datetime import datetime


import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from munch import Munch
from tensorboardX import SummaryWriter
from torch import nn

import datasets
import models
import utils

import sys
assert sys.version.startswith('3.6')
assert torch.__version__.startswith('0.4')


def adjust_learning_rate(optimizer, epoch):
    lr = cfg.optimizer.lr
    for e in cfg.optimizer.lr_decay_epochs:
        if epoch >= e:
            lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(checkpoint_dir, state, epoch):
    file_path = os.path.join(checkpoint_dir, 'checkpoint_{:08d}.pth.tar'.format(epoch))
    torch.save(state, file_path)
    return file_path


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, input, target, object_index):
        return self.loss(input, target)


class PoseCNNLoss(nn.Module):
    def __init__(self):
        super(PoseCNNLoss, self).__init__()

    def forward(self, input, target, object_index):
        position_loss = torch.abs(target[:, :3] - input[:, :3]).mean()
        orientation_loss = (1 - (input[:, 3:] * target[:, 3:]).sum(dim=1).pow(2)).mean()
        return position_loss + orientation_loss


class PointsL1Loss(nn.Module):
    def __init__(self, numpy_pcs, use_negative_qr_loss=True):
        super(PointsL1Loss, self).__init__()
        self._pcs = torch.from_numpy(np.array(numpy_pcs)).cuda()
        self.use_negative_qr_loss = use_negative_qr_loss
        if self.use_negative_qr_loss:
            self._zeros = torch.zeros(1).cuda()

    def _transformation_matrix(self, pose):
        batch_size = pose.size(0)
        H = torch.zeros(batch_size, 4, 4).cuda()
        qr, qi, qj, qk = map(lambda x: x.squeeze(1), pose[:, 3:].split(split_size=1, dim=1))
        H[:, 0, 0] = 1 - 2*(qj**2 + qk**2)
        H[:, 0, 1] = 2*(qi*qj - qk*qr)
        H[:, 0, 2] = 2*(qi*qk + qj*qr)
        H[:, 1, 0] = 2*(qi*qj + qk*qr)
        H[:, 1, 1] = 1 - 2*(qi**2 + qk**2)
        H[:, 1, 2] = 2*(qj*qk - qi*qr)
        H[:, 2, 0] = 2*(qi*qk - qj*qr)
        H[:, 2, 1] = 2*(qj*qk + qi*qr)
        H[:, 2, 2] = 1 - 2*(qi**2 + qj**2)
        H[:, :3, 3] = pose[:, :3]
        H[:, 3, 3] = 1
        return H

    def forward(self, input, target, object_index):
        batch_size = input.size(0)
        pc = torch.index_select(self._pcs, 0, object_index)
        transformed_input = torch.bmm(self._transformation_matrix(input), pc)
        transformed_target = torch.bmm(self._transformation_matrix(target), pc)
        loss = torch.abs(transformed_target - transformed_input).mean()
        if self.use_negative_qr_loss:
            zeros = self._zeros.expand(batch_size, 1)
            negative_qr_loss = torch.max(zeros, -input[:, 3]).mean()
            return loss + negative_qr_loss
        else:
            return loss


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    position_errors = utils.AverageMeter()
    orientation_errors = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.train()
    end = time.time()

    for i, (input, target, object_index) in enumerate(train_loader):
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        object_index = object_index.cuda(non_blocking=True)

        position, orientation = model(input, object_index)
        position_error = (target[:, :3] - position).pow(2).sum(dim=1).sqrt()
        orientation_error = 180.0 / np.pi * utils.batch_rotation_angle(target[:, 3:], orientation)
        output = torch.cat((position, orientation), 1)
        loss = criterion(output, target, object_index)

        position_errors.update(position_error.mean(), input.size(0))
        orientation_errors.update(orientation_error.mean(), input.size(0))
        losses.update(loss, input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.training.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Position error {position_error.val:.4f} ({position_error.avg:.4f})\t'
                  'Orientation error {orientation_error.val:.4f} ({orientation_error.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time,
                      position_error=position_errors, orientation_error=orientation_errors, loss=losses))

    return (batch_time.avg, data_time.avg, position_errors.avg, orientation_errors.avg, losses.avg)


def validate(val_loader, model, criterion):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    position_errors = utils.AverageMeter()
    orientation_errors = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target, object_index) in enumerate(val_loader):
            data_time.update(time.time() - end)

            target = target.cuda(non_blocking=True)
            object_index = object_index.cuda(non_blocking=True)

            position, orientation = model(input, object_index)
            position_error = (target[:, :3] - position).pow(2).sum(dim=1).sqrt()
            orientation_error = 180.0 / np.pi * utils.batch_rotation_angle(target[:, 3:], orientation)
            output = torch.cat((position, orientation), 1)
            loss = criterion(output, target, object_index)

            position_errors.update(position_error.mean(), input.size(0))
            orientation_errors.update(orientation_error.mean(), input.size(0))
            losses.update(loss, input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg.training.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Position error {position_error.val:.4f} ({position_error.avg:.4f})\t'
                      'Orientation error {orientation_error.val:.4f} ({orientation_error.avg:.4f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                          i, len(val_loader), batch_time=batch_time, data_time=data_time,
                          position_error=position_errors, orientation_error=orientation_errors, loss=losses))

    return (batch_time.avg, data_time.avg, position_errors.avg, orientation_errors.avg, losses.avg)


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

    cfg.arch.num_position_outputs = 3 * len(cfg.data.objects)
    cfg.arch.num_orientation_outputs = 4 * len(cfg.data.objects)
    model = models.Model(cfg.arch)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    numpy_pcs = [utils.get_pc(cfg.data.pcd_root, '{}_1000.pcd'.format(n)) for n in cfg.data.objects]
    if cfg.loss == 'l1':
        criterion = L1Loss().cuda()
    elif cfg.loss == 'posecnn':
        criterion = PoseCNNLoss().cuda()
    elif cfg.loss == 'points_simple':
        criterion = PointsL1Loss(numpy_pcs, use_negative_qr_loss=False).cuda()
    elif cfg.loss == 'points':
        criterion = PointsL1Loss(numpy_pcs).cuda()
    else:
        print('unknown loss function: {}'.format(cfg.loss))
        raise Exception

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=cfg.optimizer.lr,
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

    transform = transforms.ToTensor()
    val_dataset = datasets.RenderedPoseDataset(
        cfg.data.root, cfg.data.objects, cfg.data.val_subset_num, transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.workers, pin_memory=True)

    train_summary_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    val_summary_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))
    for epoch in range(start_epoch, cfg.training.epochs):
        lr = adjust_learning_rate(optimizer, epoch)
        train_summary_writer.add_scalar('learning_rate', lr, epoch + 1)

        # train
        if 'max_circle_size' in cfg.data:
            print('training occluded dataset with max circle size {}'.format(cfg.data.max_circle_size))
            train_dataset = datasets.OccludedRenderedPoseDataset(
                cfg.data.root, cfg.data.objects, (epoch % cfg.data.num_subsets) + 1, transform, cfg.data.max_circle_size)
        else:
            train_dataset = datasets.RenderedPoseDataset(
                cfg.data.root, cfg.data.objects, (epoch % cfg.data.num_subsets) + 1, transform)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.workers, pin_memory=True)
        (train_batch_time, train_data_time, train_position_error, train_orientation_error, train_loss) = train(
            train_loader, model, criterion, optimizer, epoch)
        train_summary_writer.add_scalar('batch_time', train_batch_time, epoch + 1)
        train_summary_writer.add_scalar('data_time', train_data_time, epoch + 1)
        train_summary_writer.add_scalar('position_error', train_position_error, epoch + 1)
        train_summary_writer.add_scalar('orientation_error', train_orientation_error, epoch + 1)
        train_summary_writer.add_scalar('loss', train_loss, epoch + 1)

        # validate
        (val_batch_time, val_data_time, val_position_error, val_orientation_error, val_loss) = validate(
            val_loader, model, criterion)
        val_summary_writer.add_scalar('batch_time', val_batch_time, epoch + 1)
        val_summary_writer.add_scalar('data_time', val_data_time, epoch + 1)
        val_summary_writer.add_scalar('position_error', val_position_error, epoch + 1)
        val_summary_writer.add_scalar('orientation_error', val_orientation_error, epoch + 1)
        val_summary_writer.add_scalar('loss', val_loss, epoch + 1)

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
