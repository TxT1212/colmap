# coding: utf-8
import os
import sys
import math
import numpy as np
import argparse
import matplotlib.pyplot as plt

sys.path.append('../')
from colmap_process.colmap_read_write_model import *
from colmap_process.colmap_keyframe_selecter import auto_read_model

'''
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
'''

def histogram_focal_length(model_folder):
    # 读colmap model
    cameras, images, points3d = auto_read_model(model_folder)
    focal_length_ratio = []
    cam_model_set = set()
    for _, value in images.items():
        cam_model = cameras[value.camera_id].model
        cam_model_set.add(cam_model)
        # 不管什么model，f/fx, fy都是第0个，且默认初始化fx = fy = 1.2 * max(width, height)
        cur_ratio = cameras[value.camera_id].params[0] / max(cameras[value.camera_id].width, cameras[value.camera_id].height)
        focal_length_ratio.append(cur_ratio)
    focal_length_ratio = np.array(focal_length_ratio)
    # print(focal_length_ratio)
    print('cam_model_set:', cam_model_set)
    plt.title('histogram_focal_length')
    plt.xlabel('ratio')
    plt.ylabel('number')
    plt.hist(focal_length_ratio)
    plt.savefig(model_folder + '/histogram_focal_length.png')
    plt.close('all')
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    histogram_focal_length(args.model_folder)
    return

if __name__ == '__main__':
    main()