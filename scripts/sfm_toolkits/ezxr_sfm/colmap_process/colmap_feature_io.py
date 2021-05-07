# coding: utf-8
import os
import sys
import numpy as np

import cv2


def read_feature_file(path_file):
    with open(path_file, "r") as fid:
      line = fid.readline()
      elems = line.split()
      keypoint_size = int(elems[0])
      keypoint_name = elems[2]
      keypoint_type = elems[3]
      keypoint_dim = int(elems[4].strip())

      descriptor = np.zeros([keypoint_size,keypoint_dim], dtype = np.uint8)
      keypoint = np.zeros([keypoint_size,2], dtype = np.int32)

      line_index = 0
      while True:
          line = fid.readline()
          if not line:
              break
          line = line.strip()
          if len(line) > 0 and line[0] != "#":
              elems = line.split()
              x = int(elems[0])
              y = int(elems[1])
              desp = np.array(tuple(map(float, elems[2:2+keypoint_dim])))
              descriptor[line_index] = desp
              keypoint[line_index] = [x,y]
              line_index += 1

    return keypoint, descriptor, keypoint_name, keypoint_type


def read_feature_file_npz(path_file):
    frame1 = np.load(path_file)
    # frame2 = np.load(path_npz2)

    # # Assert the keypoints are sorted according to the score.
    # assert np.all(np.sort(frame1['scores'])[::-1] == frame1['scores'])
    # local_descriptors = frame1['local_descriptors']
    # # WARNING: scores are not taken into account as of now.
    # des1 = frame1['local_descriptors'].astype('float32')[:1000]  

    return frame1['keypoints'], frame1['local_descriptors']
    # des2 = frame2['local_descriptors'].astype('float32')[:num_points]

def test_desp_cv(path_img):
    img = cv2.imread(path_img,0)

    # star = cv2.FeatureDetector_create("STAR")

    # # Initiate BRIEF extractor
    # brief = cv2.DescriptorExtractor_create("BRIEF")

    # # find the keypoints with STAR
    # kp = star.detect(img,None)

    # # compute the descriptors with BRIEF
    # kp, des = brief.compute(img, kp)

    orb = cv2.ORB_create()

    # find the keypoints with ORB
    kp = orb.detect(img,None)

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)  

    print(des) 

# test_desp_cv('/home/netease/Pictures/005_1.png')
#read_feature_file_npz('/media/netease/Software/Dataset/Oasis/ibl_test/npz/nikon5300a_undistor_DSC_0036.npz')
#read_feature_file('/media/netease/Software/Dataset/Oasis/ibl_mall/ibl_part/query_images_undistort/lp_IMG_0886.txt')