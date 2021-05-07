# coding: utf-8
import cv2
from cv2 import aruco
import numpy as np
import math
import yaml
import sys
import os
import python_scale_ezxr.lie_algebra as la

from python_scale_ezxr.io_tool import load_board_parameters
from python_scale_ezxr.charuco_detection import *
from colmap_process.colmap_read_write_model import *

def matcher(charuco_corners, charuco_pts_3d):
    xxx = np.zeros((3, charuco_pts_3d.shape[0]), dtype=float)
    yyy = np.zeros((3, charuco_pts_3d.shape[0]), dtype=float)
    for i in range(charuco_pts_3d.shape[0]):
        # 第三维是charuco id
        id = int(charuco_pts_3d[i, 3])
        xxx[0, i] = charuco_pts_3d[i, 0]
        xxx[1, i] = charuco_pts_3d[i, 1]
        xxx[2, i] = charuco_pts_3d[i, 2]
        # 直接通过charuco id找到对应的点
        yyy[0, i] = charuco_corners[id][0]
        yyy[1, i] = charuco_corners[id][1]
    # yyy是gt
    ret = True
    if xxx.shape[1] < 3:
        ret = False
    return ret, xxx, yyy

def write_match(colmap_pts, gt_pts, txt_path_name):
    assert(colmap_pts.shape == gt_pts.shape)
    # 申明要保存的数据的矩阵
    # 前三行是colmap pts
    # 后三行是gt pts
    save_mat = np.zeros((6, colmap_pts.shape[1]), dtype=float)
    save_mat[0:3, : ] = colmap_pts
    save_mat[3:6, : ] = gt_pts
    np.savetxt(txt_path_name, save_mat, fmt='%f')

### 输出对应点3D位置与gt3D位置的关系
def charuco_gt_match(board_param_path, charuco_pts_3d, charuco_match_folder):
    # 读标定板参数
    board_rows, board_cols, square_length, marker_length = load_board_parameters(board_param_path)
    # 生成gt
    charuco_corners = create_charuco_corners(board_rows, board_cols, square_length, marker_length)
    split_rows = 3
    split_cols = 2
    # 从model中读取point3D,和它对应的图像id
    charuco_pts_3d_dict = classify_charuco_board(charuco_pts_3d, board_rows, board_cols, split_rows, split_cols)

    print('----------------------------charuco match----------------------------')
    for key, value in charuco_pts_3d_dict.items():
        if key == 'invalid':
            print(key, 'split_charuco_board ')
            continue
        print('split_charuco_board ', key, ':')
        # 注意,这里的value只是某一块split_board上的角点的子集,colmap建图所用的图像上不一定每个角点都检测到了
        value = np.array(value)
        ret, colmap_pts, gt_pts = matcher(charuco_corners, value)
        
        if ret:
            txt_path_name = charuco_match_folder + '/' + key + '.txt'
            write_match(colmap_pts, gt_pts, txt_path_name)
    # return charuco_pts_3d_dict, charuco_corners


def main():
    if len(sys.argv) != 4:
        print('charuco_match [board config yaml] [path to charuco points3d folder] [path to charuco match folder].')
        return
    # 读标定板参数
    board_rows, board_cols, square_length, marker_length = load_board_parameters(sys.argv[1])
    # 生成gt
    charuco_corners = create_charuco_corners(board_rows, board_cols, square_length, marker_length)
    # 读取charuco corners的三维点
    charuco_points3d_folder = sys.argv[2]
    charuco_match_folder = sys.argv[3]
    if charuco_points3d_folder[-1] != '/':
        charuco_points3d_folder = charuco_points3d_folder + '/'
    if charuco_match_folder[-1] != '/':
        charuco_match_folder = charuco_match_folder + '/'
    charuco_points3d_path_name = charuco_points3d_folder + 'charuco_point_3d.txt'
    charuco_pts_3d_all = np.loadtxt(charuco_points3d_path_name, dtype=float)
    # 对charuco corners的三维点进行split board分类
    # 有时候charuco误检,charuco_id大概率是错的,invalid,跳过即可
    split_rows = 3
    split_cols = 2
    charuco_pts_3d_dict = classify_charuco_board(charuco_pts_3d_all, board_rows, board_cols, split_rows, split_cols)
    print('----------------------------charuco match----------------------------')
    for key, value in charuco_pts_3d_dict.items():
        if key == 'invalid':
            print(key, 'split_charuco_board ')
            continue
        print('split_charuco_board ', key, ':')
        # 注意,这里的value只是某一块split_board上的角点的子集,colmap建图所用的图像上不一定每个角点都检测到了
        value = np.array(value)
        ret, colmap_pts, gt_pts = matcher(charuco_corners, value)
        if ret:
            txt_path_name = charuco_match_folder + key + '.txt'
            write_match(colmap_pts, gt_pts, txt_path_name)
    print('All done!')

if __name__ == '__main__':
    main()
