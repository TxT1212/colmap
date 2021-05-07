# coding: utf-8
import sys
import sqlite3
import math
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append('../')
from python_scale_ezxr.scale_restoration import umeyama_alignment
from python_scale_ezxr.visualization import visualize_pts_3d_two

def error_of_sim3_align(src_path, tgt_path, ignore_z, is_sim3):
    '''
    假设两个点集行数相同，每行都是对应
    '''
    src_data = np.loadtxt(src_path)
    src = src_data[:, 1:4]
    src = src.T
    tgt_data = np.loadtxt(tgt_path)
    tgt = tgt_data[:, 1:4]
    tgt = tgt.T
    if ignore_z:
        src[2, : ] = 0.0
        tgt[2, : ] = 0.0
    rmat, tvec, scale = umeyama_alignment(src, tgt, is_sim3)
    src_transformed = np.matmul(rmat, scale * src) + tvec.reshape(3,1)
    delta = src_transformed - tgt
    delta_norm = np.linalg.norm(delta, axis=0)
    delta_norm = sorted(delta_norm)
    mean_delta_norm = np.mean(delta_norm)
    median_delta_norm = np.median(delta_norm)
    print('scale = ', scale)
    print('mean = ', mean_delta_norm)
    print('median = ', median_delta_norm)
    return delta_norm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', required=True)
    parser.add_argument('--tgt_path', required=True)
    parser.add_argument('--ignore_z', action='store_true')
    parser.add_argument('--is_sim3', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    error_of_sim3_align(args.src_path, args.tgt_path, args.ignore_z, args.is_sim3)

if __name__ == "__main__":
    main()