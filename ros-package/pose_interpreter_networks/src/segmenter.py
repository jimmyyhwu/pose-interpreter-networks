#!/usr/bin/env python

import time

import cv2
import message_filters
import numpy as np
import rospy
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from cv_bridge import CvBridge, CvBridgeError
from matplotlib import colors
from munch import Munch
from sensor_msgs.msg import Image, CameraInfo

import segmentation_models


class Segmenter:

    def __init__(self):
        # model
        segm_cfg = Munch.fromDict(rospy.get_param('segmentation'))
        segm_model = segmentation_models.DRNSeg(segm_cfg.arch, segm_cfg.data.classes, None, pretrained=True)
        segm_model = torch.nn.DataParallel(segm_model).cuda()
        cudnn.benchmark = True
        resume_path = rospy.get_param('segmentation_checkpoint')
        checkpoint = torch.load(resume_path)
        segm_model.load_state_dict(checkpoint['state_dict'])
        segm_model.eval()
        rospy.loginfo("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
        self.segm_model = segm_model

        # preprocessing
        self.transform = transforms.ToTensor()

        # visualization
        self.segmentation_alpha = rospy.get_param('segmentation_alpha')
        self.cmap = 255.0 * np.array(map(colors.to_rgb, rospy.get_param('object_colors')))

        # ros
        self.image_sub = message_filters.Subscriber(rospy.get_param('image_sub_topic'), Image)
        self.info_sub = message_filters.Subscriber(rospy.get_param('info_sub_topic'), CameraInfo)
        # note: using message_filters.TimeSynchronizer rather than just rospy.Subscriber
        # significantly improves the latency of visualization in rviz, unclear why...
        self.ts = message_filters.TimeSynchronizer([self.image_sub, self.info_sub], queue_size=1)
        self.ts.registerCallback(self.callback)
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher(rospy.get_param('segmentation_pub_topic'), Image, queue_size=1)

    def callback(self, image, camera_info):
        start_time = time.time()

        # get image from ros format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr(e)
        np_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # forward pass
        with torch.no_grad():
            input = self.transform(np_image).unsqueeze(0)
            output = self.segm_model(input).max(1)[1]
            prediction = output.cpu().squeeze(0).numpy()

        # visualization
        np_image = (1 - self.segmentation_alpha) * np_image + self.segmentation_alpha * self.cmap[prediction]

        # convert back to ros format
        cv_image = cv2.cvtColor(np_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, 'bgr8'))
        except CvBridgeError as e:
            rospy.logerr(e)

        rospy.loginfo('callback time: {}ms'.format(int(1000 * (time.time() - start_time))))


def main():
    rospy.init_node('segmenter', anonymous=True)
    segmenter = Segmenter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
