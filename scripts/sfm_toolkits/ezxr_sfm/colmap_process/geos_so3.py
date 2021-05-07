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
def float2str(a):
    return str(format(a, '.4f'))

def read_geo(file_path_name):
    with open(file_path_name, 'r') as file:
        data = file.read().split('\n')
        # delete empty line
        if (len(data[-1]) < 1):
            data = data[0:-1]
        geo_dict = {}
        for item in data:
            element = item.split(' ')
            cam_name = 'cam/' + element[0].split('/')[-1]
            geo_dict[cam_name] = np.array([float(element[1]), float(element[2]), float(element[3])])
        # print(geo_dict)
    return geo_dict

def so3_transform_and_write_geo(file_path_name, data, names, r, t, s):
    assert(data.shape[1] == len(names))

    src_new = s * np.matmul(r, data) + t.reshape(3,1)

    geo_file = open(file_path_name,'w')
    for idx in range(data.shape[1]):
        geo_line = names[idx] + ' ' + float2str(src_new[0,idx]) + ' ' + float2str(src_new[1,idx]) + ' ' + float2str(src_new[2,idx]) + '\n'
        geo_file.write(geo_line) 
    geo_file.close()

def write_report(file_path_name, args, statistic, origin_r, t, s, threshold):
    geo_file = open(file_path_name,'w')
    geo_line = '# src_geos: ' + args.src_geos + '\n'
    geo_file.write(geo_line)
    geo_line = '# tgt_geos: ' + args.tgt_geos + '\n'
    geo_file.write(geo_line)
    geo_line = '# rmse mean, median, std, match_num, inlier_ratio, inlier_t(m): \n# ' + \
            float2str(statistic[0]) + ', ' + float2str(statistic[1]) + ', ' + \
            float2str(statistic[2]) + ', ' + str(statistic[3]) + ', ' + \
            float2str(statistic[4]) + ', ' + float2str(statistic[5]) + ', ' + \
            str(threshold) + '\n'
    geo_file.write(geo_line)
    geo_line = '# se3/sim3 transform from src to tgt: \n'
    geo_file.write(geo_line)
    r = origin_r * s
    geo_line = float2str(r[0, 0]) + ' ' + float2str(r[0, 1]) + ' ' + float2str(r[0, 2]) + ' ' + float2str(t[0]) + '\n' + \
                float2str(r[1, 0]) + ' ' + float2str(r[1, 1]) + ' ' + float2str(r[1, 2]) + ' ' + float2str(t[1]) + '\n' + \
                float2str(r[2, 0]) + ' ' + float2str(r[2, 1]) + ' ' + float2str(r[2, 2]) + ' ' + float2str(t[2]) + '\n' + \
                '0.0 0.0 0.0 1.0'
    geo_file.write(geo_line)
    geo_file.close()
    return

def match_position(geo_dict_src, geo_dict_tgt):
    two_positions = []
    names = []
    if len(geo_dict_src) < len(geo_dict_tgt):
        for name, position in  geo_dict_src.items():
            if name in geo_dict_tgt: # match success
                pos_tgt = geo_dict_tgt[name]
                two_pos = [position[0], position[1], position[2], pos_tgt[0], pos_tgt[1], pos_tgt[2]]
                two_positions.append(two_pos)
                names.append(name)
    else:
        for name, position in  geo_dict_tgt.items():
            if name in geo_dict_src:
                pos_src = geo_dict_src[name]
                two_pos = [pos_src[0], pos_src[1], pos_src[2], position[0], position[1], position[2]]
                two_positions.append(two_pos)
                names.append(name)
    
    two_positions = np.array(two_positions)
    two_positions = two_positions.transpose()
    return two_positions, names

def so3_transform_statistic(r, t, s, src, tgt, threshold):
    src_transformed = s * np.matmul(r, src) + t.reshape(3,1)
    delta = src_transformed - tgt
    # print(delta.shape)
    delta_norm = np.linalg.norm(delta, axis=0)

    inlier_idxs = np.where(delta_norm < threshold)[0]
    inlier_norm = delta_norm[inlier_idxs]
    # print(inlier_norm)

    if len(inlier_norm) == 0:
        return inlier_norm, np.array([np.inf, np.inf, np.inf])
    rmse = np.sqrt( np.mean( np.power(inlier_norm, 2) ) )
    mean_error = np.mean(inlier_norm)
    median_error = np.median(sorted(inlier_norm))
    std_error = np.std(inlier_norm)
    # print('pt_num = ', src.shape[1], 'mean = ', mean_error, ', median = ', median_error, 'std = ', std_error)
    statistic = np.array([rmse, mean_error, median_error, std_error, len(inlier_norm), len(inlier_norm) * 1.0 / len(delta_norm)])
    return inlier_idxs, statistic

def random_partition(n, n_data):
    """return n random rows of data and the other len(data) - n rows"""
    all_idxs = np.arange(n_data.shape[1]) #获取n_data下标索引
    np.random.shuffle(all_idxs) #打乱下标索引
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2

def so3_transform_ransac(data, n, k, t, is_sim3):
    """
    参考:http://scipy.github.io/old-wiki/pages/Cookbook/RANSAC
    伪代码:http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182
    输入:
        data - 样本点
        n - 生成模型所需的最少样本点
        k - 最大迭代次数
        t - 阈值:作为判断点满足模型的条件
    输出:
        bestfit - 最优拟合解（返回nil,如果未找到）
    """
    iterations = 0
    best_r = None
    best_t = None
    best_s = None
    best_inlier_idxs = []
    best_statistic = np.array([np.inf, np.inf, np.inf])
    while iterations < k:
        # ------random------
        maybe_idxs, _ = random_partition(n, data)
        maybe_inliers = data[:, maybe_idxs]
        # ------fit   ------
        maybe_r, maybe_t, maybe_s = umeyama_alignment(maybe_inliers[0:3, : ], maybe_inliers[3:6, : ], is_sim3)
        # ------test  ------
        test_inlier_idxs, test_statistic = so3_transform_statistic(maybe_r, maybe_t, maybe_s, data[0:3, : ], data[3:6, : ], t)
        # ------compare------
        if len(test_inlier_idxs) > len(best_inlier_idxs):# 内点个数更多,直接更新最佳模型
            best_r = maybe_r
            best_t = maybe_t
            best_s = maybe_s
            best_inlier_idxs = test_inlier_idxs
            best_statistic = test_statistic
            print('A->best_statistic:', best_statistic)
        elif len(test_inlier_idxs) == len(best_inlier_idxs): # 内点个数相当,要进一步比较误差
            if (test_statistic[0] < best_statistic[0]):
                best_r = maybe_r
                best_t = maybe_t
                best_s = maybe_s
                best_inlier_idxs = test_inlier_idxs
                best_statistic = test_statistic
                print('B->best_statistic:', best_statistic)
        iterations = iterations + 1
        
    if best_r is None or best_t is None:
        raise ValueError("did't meet fit acceptance criteria")
    # use all inliers refine the se3/sim3
    best_inliers = data[:, best_inlier_idxs]
    best_r, best_t, best_s = umeyama_alignment(best_inliers[0:3, : ], best_inliers[3:6, : ], is_sim3)
    best_inlier_idxs, best_statistic = so3_transform_statistic(best_r, best_t, best_s, data[0:3, : ], data[3:6, : ], t)
    print('C->best_statistic:', best_statistic)
    return best_r, best_t, best_s, best_inlier_idxs, best_statistic

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_geos', required=True)
    parser.add_argument('--tgt_geos', required=True)
    # parser.add_argument('--transformed_geos', required=True)
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--is_sim3', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    np.set_printoptions(suppress=True)
    report_path_name = args.src_geos[0:-4] + '_report.txt'
    transformed_geos_path = args.src_geos[0:-4] + '_new.txt'
    # read geos
    geo_dict_src = read_geo(args.src_geos)
    geo_dict_tgt = read_geo(args.tgt_geos)
    # match
    data, names = match_position(geo_dict_src, geo_dict_tgt)
    # so3_ransac
    best_r, best_t, best_s, best_inlier_idxs, best_statistic = so3_transform_ransac(data, 3, 1000, args.threshold, args.is_sim3)
    # visualize
    # src_new = best_s * np.matmul(best_r, data[0:3,:]) + best_t.reshape(3,1)
    # visualize_pts_3d_two(src_new[0:3,:].transpose(), data[3:6,:].transpose())
    # transform
    so3_transform_and_write_geo(transformed_geos_path, data[0:3, best_inlier_idxs], np.array(names)[best_inlier_idxs], best_r, best_t, best_s)
    write_report(report_path_name, args, best_statistic, best_r, best_t, best_s, args.threshold)
    return

if __name__ == '__main__':
    main()