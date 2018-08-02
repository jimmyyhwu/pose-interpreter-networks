#!/usr/bin/env python

import os
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
from visualization_msgs.msg import Marker, MarkerArray

import pose_estimation_models
import segmentation_models
import models


def create_marker(frame_id, object_id, stl_path, color, alpha=0.75, duration=0.1):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.id = object_id
    marker.type = marker.MESH_RESOURCE
    marker.action = marker.ADD
    marker.lifetime = rospy.Duration.from_sec(duration)
    marker.mesh_resource = 'file://' + stl_path
    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 1.0
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = alpha
    return marker


class PoseEstimator:

    def __init__(self):
        self.pub_segm = rospy.get_param('pub_segmentation')

        # segmentation model
        segm_cfg = Munch.fromDict(rospy.get_param('segmentation'))
        segm_model = segmentation_models.DRNSeg(segm_cfg.arch, segm_cfg.data.classes, None, pretrained=True)
        segm_model = torch.nn.DataParallel(segm_model).cuda()
        segm_resume_path = rospy.get_param('segmentation_checkpoint')
        segm_checkpoint = torch.load(segm_resume_path)
        segm_model.load_state_dict(segm_checkpoint['state_dict'])
        rospy.loginfo("=> loaded checkpoint '{}' (epoch {})".format(segm_resume_path, segm_checkpoint['epoch']))

        # pose model
        pose_cfg = Munch.fromDict(rospy.get_param('pose_estimation'))
        pose_model = pose_estimation_models.Model(pose_cfg.arch)
        pose_model = torch.nn.DataParallel(pose_model).cuda()
        pose_resume_path = rospy.get_param('pose_estimation_checkpoint')
        pose_checkpoint = torch.load(pose_resume_path)
        pose_model.load_state_dict(pose_checkpoint['state_dict'])
        rospy.loginfo("=> loaded checkpoint '{}' (epoch {})".format(pose_resume_path, pose_checkpoint['epoch']))

        # end-to-end model
        object_ids = 0
        object_names_to_ids = {object_name: i for i, object_name in enumerate(segm_cfg.data.class_names)}
        object_ids = [object_names_to_ids[object_name] for object_name in pose_cfg.data.objects]
        model = models.EndToEndModel(segm_model, pose_model, pose_cfg.data.objects, object_ids)
        self.model = torch.nn.DataParallel(model).cuda()
        self.model.eval()
        cudnn.benchmark = True

        # preprocessing
        self.transform = transforms.ToTensor()

        # visualization
        cmap = map(colors.to_rgb, rospy.get_param('object_colors'))
        if self.pub_segm:
            self.segm_alpha = rospy.get_param('segmentation_alpha')
            self.segm_cmap = 255.0 * np.array(cmap)
        frame_id = rospy.get_param('frame_id')
        stl_root = rospy.get_param('stl_root')
        marker_alpha = rospy.get_param('marker_alpha')
        duration = rospy.get_param('marker_duration')
        self.markers = {}
        for object_name in pose_cfg.data.objects:
            stl_path = os.path.join(stl_root, '{}.stl'.format(object_name))
            object_id = object_names_to_ids[object_name]
            self.markers[object_name] = create_marker(frame_id, object_id, stl_path, cmap[object_id], marker_alpha, duration)

        # ros
        self.image_sub = message_filters.Subscriber(rospy.get_param('image_sub_topic'), Image)
        self.info_sub = message_filters.Subscriber(rospy.get_param('info_sub_topic'), CameraInfo)
        # note: using message_filters.TimeSynchronizer rather than just rospy.Subscriber
        # significantly improves the latency of visualization in rviz, unclear why...
        self.ts = message_filters.TimeSynchronizer([self.image_sub, self.info_sub], queue_size=1)
        self.ts.registerCallback(self.callback)
        self.bridge = CvBridge()
        if self.pub_segm:
            self.segm_pub = rospy.Publisher(rospy.get_param('segmentation_pub_topic'), Image, queue_size=1)
        self.pose_pub = rospy.Publisher(rospy.get_param('pose_estimate_pub_topic'), MarkerArray, queue_size=1)

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
            segm, object_names, positions, orientations = self.model(input)

        # visualize objects
        marker_array = MarkerArray()
        for i, object_name in enumerate(object_names):
            position = positions[i]
            orientation = orientations[i]
            marker = self.markers[object_name]
            marker.pose.position.x = position[0]
            marker.pose.position.y = position[1]
            marker.pose.position.z = position[2]
            marker.pose.orientation.w = orientation[0]
            marker.pose.orientation.x = orientation[1]
            marker.pose.orientation.y = orientation[2]
            marker.pose.orientation.z = orientation[3]
            marker_array.markers.append(marker)

        # segmentation visualization
        if self.pub_segm:
            segm_image = (1 - self.segm_alpha) * np_image + self.segm_alpha * self.segm_cmap[segm]
            segm_cv_image = cv2.cvtColor(segm_image.astype(np.uint8), cv2.COLOR_RGB2BGR)

        try:
            self.pose_pub.publish(marker_array)
            if self.pub_segm:
                self.segm_pub.publish(self.bridge.cv2_to_imgmsg(segm_cv_image, 'bgr8'))
        except CvBridgeError as e:
            rospy.logerr(e)

        rospy.loginfo('callback time: {}ms'.format(int(1000 * (time.time() - start_time))))


def main():
    rospy.init_node('pose_estimator', anonymous=True)
    pose_estimator = PoseEstimator()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
