#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import six
import time
import cv2
import drawing

from chainer import cuda

import alexnet
import log_initializer
import model_io
import normalizers

# logging
from logging import getLogger, DEBUG, INFO

log_initializer.setFmt()
log_initializer.setRootLevel(DEBUG)
logger = getLogger(__name__)

# for Python 2.x
try:
    tmp = FileNotFoundError
except NameError:
    FileNotFoundError = IOError

JOINT_MAP = {
    'lsho': 0,  # L_Shoulder
    'lelb': 1,  # L_Elbow
    'lwri': 2,  # L_Wrist
    'rsho': 3,  # R_Shoulder
    'relb': 4,  # R_Elbow
    'rwri': 5,  # R_Wrist
    'lhip': 6,  # L_Hip
    'rhip': 7,  # R_Hip
    'head': 8,  # Head
}


class PoseDetector(object):
    def __init__(self, detector=None):
        # initialize arguments
        stage = 0
        GPU = 0

        cuda.check_cuda_available()
        logger.info('GPU mode (%d) (stage: %d)', GPU, stage)
        self.xp = cuda.cupy

        if detector is None:
            # Face Detector
            self.detector = normalizers.FaceDetector([
                "cascade/haarcascade_frontalface_alt.xml",
                "cascade/lbpcascade_frontalface.xml"
            ])
        # Pose Normalizer
        self.normalizer = normalizers.FaceBasedPoseNormalizer()
        facial_normalizer_path = "data/facial_normalizer.npy"
        ret = self.normalizer.load(facial_normalizer_path)  # load
        self.model_exist_map = {}
        self.model_cache_map = {}

    def use_model_single(self, stage_cnt, joint_idx, model, img,  pre_joint=None):
        width, height = alexnet.IMG_WIDTH, alexnet.IMG_HEIGHT
        xp_img_shape = self.xp.asarray([width, height], dtype=np.float32)

        # Normalize
        if stage_cnt == 0:
            # facial_rect = detector.detect_joint_valid_face(img, teacher)
            facial_rect = self.detector.detect_biggest_face(img)
            if facial_rect is None:
                logger.info('Failed to detect face')
                return None
            mat = self.normalizer.calc_matrix(width, height, facial_rect)
        else:
            center = pre_joint[joint_idx]
            mat = normalizers.calc_cropping_matrix(width, height, center, pre_joint,
                                                   sigma=1.5)
        img = normalizers.transform_img(img, mat, width, height)

        # img -> imgs
        imgs = img.reshape((1,) + img.shape)
        # numpy -> chainer variable
        x = normalizers.conv_imgs_to_chainer(self.xp, imgs, train=False)
        # Use model
        pred = model(x)
        # chainer variable -> numpy
        if stage_cnt == 0:
            joint_scale_mode = '+'
        else:
            joint_scale_mode = '+-'
        pred_joints = normalizers.conv_joints_from_chainer(self.xp, pred,
                                                           xp_img_shape,
                                                           joint_scale_mode)

        # joints -> joint
        pred_joint = pred_joints[0]
        inv_mat = np.linalg.inv(mat)

        # Denormalize
        if stage_cnt == 0:
            pred_joint = normalizers.transform_joint(pred_joint, inv_mat)
            return pred_joint
        else:
            assert (len(pred_joint) == 1)
            diff_pt = pred_joint[0]
            diff_pt = normalizers.transform_joint_pt(diff_pt, inv_mat,
                                                     translate=False)
            return diff_pt

    def load_img(self, path):
        img = cv2.imread(path)
        if img is None:
            logger.error('Failed to load image (%s)', path)
            return None
        img = img.astype(np.float32) / 255.0
        return img

    def load_model(self, stage_id, joint_id):
        key = str(stage_id) + '|' + str(joint_id)
        if key in self.model_exist_map:
            return self.model_cache_map[key]
        self.model_exist_map[key] = True
        try:
            model = model_io.load_best_model(stage_id, joint_id)
            self.model_cache_map[key] = model
        except FileNotFoundError:
            logger.info('Failed to load')
            self.model_cache_map[key] = None
        return self.model_cache_map[key]

    def detect(self, raw_img):
        # Load first model and use
        first_model = self.load_model(0, -1)
        pred_joint = self.use_model_single(0, -1, first_model, raw_img)
        if pred_joint is None:
            return

        results = [pred_joint]

        # Subsequent models
        for joint_idx in six.moves.xrange(len(JOINT_MAP)):
            stage_cnt = 0
            while True:
                stage_cnt += 1
                # Try to load next stage model
                model = self.load_model(stage_cnt, joint_idx)
                if model is None:
                    break
                # Use model
                diff_pt = self.use_model_single(stage_cnt, joint_idx, model, raw_img,
                                                pre_joint=results[stage_cnt - 1])
                # Create new result
                if len(results) <= stage_cnt:
                    next_joint = np.array(results[stage_cnt - 1], copy=True)
                    results.append(next_joint)
                # Apply
                results[stage_cnt][joint_idx] += diff_pt

            # Show
            for stage_cnt, result in enumerate(results):
                # single -> multi
                imgs = raw_img.reshape((1,) + raw_img.shape)
                joints = result.reshape((1,) + result.shape)
                for img, joint in zip(imgs, joints):
                    self.draw_detect(img, joint)
            cv2.waitKey(3300)

    def draw_detect(self, img, joints):
        drawing.draw_joint(img, joints)
        cv2.imshow("" + str(time.time()), img)


pose_detector = PoseDetector()

raw_img = pose_detector.load_img('scripts/img/test2.jpg')
pose_detector.detect(raw_img)
