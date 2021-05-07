# coding: utf-8
import cv2
from cv2 import aruco
import numpy as np
import math
import yaml
import sys
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares

import python_scale_ezxr.lie_algebra as la

from python_scale_ezxr.io_tool import load_board_parameters
from python_scale_ezxr.charuco_detection import *
from python_scale_ezxr.visualization import *
from colmap_process.colmap_read_write_model import *

def umeyama_alignment(x, y, with_scale=False):
    """
    R*x + t = y
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """
    if x.shape != y.shape:
        print("data matrices must have the same shape")
        sys.exit(0)

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c

def sim3_transform(xxx, yyy):
    '''
    umeyama_alignment(x, y, with_scale=False)
    算出来的transform是from x to y
    '''
    # 把colmap的点转到gt,原因是:gt作为参考,不能尺度缩放
    r, t, c = umeyama_alignment(xxx, yyy, True)
    # 把r,t求逆,让transform是gt到colmap的点
    r1 = r.transpose()
    t1 = -np.matmul(r1, t.reshape(3,1))
    # 把gt转到colmap地图坐标系
    yyy_transformed = np.matmul(r1, yyy) + t1.reshape(3,1)
    # colmap的点是需要尺度缩放的
    xxx_scaled = xxx * c
    # 计算误差
    delta = xxx_scaled - yyy_transformed
    mean_error = np.sum(np.linalg.norm(delta, axis=0)) / xxx.shape[1]
    #visual_1
    #visualize_pts_3d_two(xxx_scaled.transpose(), yyy_transformed.transpose())
    # 注意,这里返回的是gt到colmap坐标系的transform
    # 尺度因子则是作用colmap的点,因为它们不是真实尺度
    ret = True
    if mean_error > 0.015:
        ret = False
        print('Failed! mean error too big! skip this split_board...')
    return ret, mean_error, r1, t1, c

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def optimize_n_poses_and_scale(init_params, scale):
    '''
    init_params是个list,每个元素是(rot, trans, colmap_pts, gt_pts)
    其中rot * gt_pts + trans = colmap_pts * scale
    '''
    n_poses = len(init_params)
    init_x = np.zeros(n_poses * 7 + 1)
    # 最后一位放尺度
    init_x[-1] = scale
    # 构建符合格式的待优化参数
    for i in range(n_poses):
        qvec = rotmat2qvec(init_params[i][0])
        tvec = init_params[i][1]
        init_x[i * 7 + 0] = qvec[0]
        init_x[i * 7 + 1] = qvec[1]
        init_x[i * 7 + 2] = qvec[2]
        init_x[i * 7 + 3] = qvec[3]
        init_x[i * 7 + 4] = tvec[0]
        init_x[i * 7 + 5] = tvec[1]
        init_x[i * 7 + 6] = tvec[2]
    # 构建残差函数
    # 优化目标是:n个pose和1个scale
    # 优化残差是:三维点的平均误差
    def residual_function(x):
        n_poses = int( ( len(x) - 1 ) / 7 )
        sum_error = 0.0
        for i in range(n_poses):
            qvec = x[i * 7 + 0 : i * 7 + 4]
            rotmat = qvec2rotmat(qvec)
            tvec = x[i * 7 + 4 : i * 7 + 7]
            colmap_pts = init_params[i][2]
            gt_pts = init_params[i][3]
            colmap_pts_scaled = colmap_pts * x[-1]
            gt_pts_transformed = np.matmul(rotmat, gt_pts) + tvec.reshape(3,1)
            delta = colmap_pts_scaled - gt_pts_transformed
            error = np.sum(np.linalg.norm(delta, axis=0))
            sum_error = sum_error + error
        return sum_error
    print('optimizing...')
    res = least_squares(residual_function, init_x, jac='3-point', loss='linear', verbose=1, max_nfev=100000)
    return res.x

def compute_scale(match_folders, report_path):
    report_file = open(report_path,'w')
    #match_folder_names = os.listdir( match_folder )
    init_params = []
    scale_list = []
    txt_name_list = []
    report_file.write('----------------------------init scale----------------------------\n')
   # subdirs = find_sub_dirs(image_parent_folder)
    for match_folder_name in match_folders:
        current_folder = match_folder_name + '/'
        txt_names = os.listdir( current_folder )
        for txt_name in txt_names:
            txt_path_name = current_folder + txt_name
            match_pts = np.loadtxt(txt_path_name, dtype=float)
            xxx = match_pts[0:3, : ]
            yyy = match_pts[3:6, : ]
            ret, mean_error, rot, trans, scale = sim3_transform(xxx, yyy)
            detailed_str = 'board id = ' + txt_name[0:-4] + ', mean error(unit:m) = ' + str(mean_error) + ', point number = '+ str(xxx.shape[1]) + ', scale = ' + str(scale)
            report_str = ', failed! '
            if ret:
                init_params.append( (rot, trans, xxx, yyy) )
                scale_list.append(scale)
                report_str = ', successful! '
                txt_name_list.append(txt_name[0:-4])
            report_str = detailed_str + report_str + '\n'
            report_file.write(report_str)
    if len(scale_list) == 0:
        print('failed! no board detected!')
        report_file.close()
        return -1
    # 对scale进行分析
    scale_list = np.array(scale_list)
    scale_mean = np.mean(scale_list)
    scale_std = np.std(scale_list)
    report_str = 'scale mean = '+ str(scale_mean) + ', std = ' + str(scale_std) + '\n'
    report_file.write(report_str)
    report_file.close()
    return scale_mean
    '''
    report_file = open(report_path,'a')
    report_file.write('----------------------------opt scale----------------------------\n')
    opt_x = optimize_n_poses_and_scale(init_params, scale_mean)
    # 二次验证优化结果要比线性结果好
    print('opted scale = ', opt_x[-1])
    for i in range(len(init_params)):
        qvec = opt_x[i * 7 + 0 : i * 7 + 4]
        rotmat = qvec2rotmat(qvec)
        tvec = opt_x[i * 7 + 4 : i * 7 + 7]
        colmap_pts = init_params[i][2]
        gt_pts = init_params[i][3]
        colmap_pts_scaled = colmap_pts * opt_x[-1]
        gt_pts_transformed = np.matmul(rotmat, gt_pts) + tvec.reshape(3,1)
        delta = colmap_pts_scaled - gt_pts_transformed
        mean_error = np.sum(np.linalg.norm(delta, axis=0)) / gt_pts.shape[1]
        detailed_str = 'board id = ' + txt_name_list[i] + ', mean error(unit:m) = ' + str(mean_error) + ', point number = ' + str(gt_pts.shape[1]) + '\n'
        report_file.write(detailed_str)
        print(detailed_str)
    scale_str = 'opted scale = ' + str(opt_x[-1]) + '\n'
    print(scale_str)
    report_file.write(scale_str)
    report_file.close()
    return opt_x[-1]
    '''

def main():
    if len(sys.argv) != 2:
        print('scale_restoration [path to colmap project folder].')
        return
    match_folder = sys.argv[1] + '/images_charuco/match/'
    report_path_name = sys.argv[1] + '/images_charuco/report.txt'
    report_file = open(report_path_name,'w')
    match_folder_names = os.listdir( match_folder )
    init_params = []
    scale_list = []
    txt_name_list = []
    report_file.write('----------------------------init scale----------------------------\n')
    for match_folder_name in match_folder_names:
        current_folder = match_folder + match_folder_name + '/'
        txt_names = os.listdir( current_folder )
        for txt_name in txt_names:
            txt_path_name = current_folder + txt_name
            match_pts = np.loadtxt(txt_path_name, dtype=float)
            xxx = match_pts[0:3, : ]
            yyy = match_pts[3:6, : ]
            ret, mean_error, rot, trans, scale = sim3_transform(xxx, yyy)
            detailed_str = 'board id = ' + txt_name[0:-4] + ', mean error(unit:m) = ' + str(mean_error) + ', point number = '+ str(xxx.shape[1]) + ', scale = ' + str(scale)
            report_str = ', failed! '
            if ret:
                init_params.append( (rot, trans, xxx, yyy) )
                scale_list.append(scale)
                report_str = ', successful! '
                txt_name_list.append(txt_name[0:-4])
            report_str = detailed_str + report_str + '\n'
            report_file.write(report_str)
    if len(scale_list) == 0:
        print('failed! no board detected!')
        report_file.close()
        return
    # 对scale进行分析
    scale_list = np.array(scale_list)
    scale_mean = np.mean(scale_list)
    scale_std = np.std(scale_list)
    report_str = 'scale mean = '+ str(scale_mean) + ', std = ' + str(scale_std) + '\n'
    report_file.write(report_str)
    report_file.close()
    report_file = open(report_path_name,'a')
    report_file.write('----------------------------opt scale----------------------------\n')
    opt_x = optimize_n_poses_and_scale(init_params, scale_mean)
    # 二次验证优化结果要比线性结果好
    print('opted scale = ', opt_x[-1])
    for i in range(len(init_params)):
        qvec = opt_x[i * 7 + 0 : i * 7 + 4]
        rotmat = read_write_model.qvec2rotmat(qvec)
        tvec = opt_x[i * 7 + 4 : i * 7 + 7]
        colmap_pts = init_params[i][2]
        gt_pts = init_params[i][3]
        colmap_pts_scaled = colmap_pts * opt_x[-1]
        gt_pts_transformed = np.matmul(rotmat, gt_pts) + tvec.reshape(3,1)
        delta = colmap_pts_scaled - gt_pts_transformed
        mean_error = np.sum(np.linalg.norm(delta, axis=0)) / gt_pts.shape[1]
        detailed_str = 'board id = ' + txt_name_list[i] + ', mean error(unit:m) = ' + str(mean_error) + ', point number = ' + str(gt_pts.shape[1]) + '\n'
        report_file.write(detailed_str)
        print(detailed_str)
    scale_str = 'opted scale = ' + str(opt_x[-1]) + '\n'
    print(scale_str)
    report_file.write(scale_str)
    report_file.close()
    print('All done!')

if __name__ == '__main__':
    main()
