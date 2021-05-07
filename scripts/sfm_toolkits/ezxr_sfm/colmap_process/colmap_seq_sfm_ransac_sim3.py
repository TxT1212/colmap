# coding: utf-8
import math
import sys
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append('../')
from colmap_process.colmap_keyframe_selecter import *
from colmap_process.colmap_camera_models import * 
from colmap_process.geos_so3 import random_partition
from python_scale_ezxr.scale_restoration import umeyama_alignment
'''
摘自colmap->src->base->similarity_transform.cc
函数TransformPose
// Projection matrix P1 projects 3D object points to image plane and thus to
// 2D image points in the source coordinate system:
//    x' = P1 * X1
// 3D object points can be transformed to the destination system by applying
// the similarity transformation S:
//    X2 = S * X1
// To obtain the projection matrix P2 that transforms the object point in the
// destination system to the 2D image points, which do not change:
//    x' = P2 * X2 = P2 * S * X1 = P1 * S^-1 * S * X1 = P1 * I * X1
// and thus:
//    P2' = P1 * S^-1
// Finally, undo the inverse scaling of the rotation matrix:
//    P2 = s * P2'
-------------------------------------------------------------------------------
bad news:
(用4个点代表pose)，在sim3的时候行不通
因为loc全局坐标系下的单位坐标轴， 在乘以scale后， 会拉大或者缩小坐标轴的值， 
而locmap全局坐标系下的单位坐标轴不变， 影响残差度量;
(用4个点代表pose)，只能在SE3的时候用

while (1):
    1. 根据轨迹align得到sim3

    2. 把loc全局坐标系下的3d点, 通过sim3变换到locmap全局坐标系下
    X_global-locmap = sim3_global-loc_to_global-locmap * X_global-loc

    2. 用locmap里的pose, 投影locmap全局坐标系下的3d点, 到相机坐标系的归一化3d点
    X_local-locmap = T_global-locmap_to_local-locmap * X_global-locmap

    3. 用loc-sfm模型的相机内参投影相机坐标系的归一化3d点(控制变量)
    x_cam-in-loc = P_cam-in-loc * X_local-locmap

    4. 计算原xy和新投影的xy^之间的像素残差, 根据threshold计算内点数

    5. 根据最大的内点数, 和内点数相同的情况下, 最小的像素误差, 保存最好的sim3


通过上述算法逻辑, 我们不需要对pose进行sim3变换, 只需要对点进行sim3变换
'''

def sim3_visualize_pts_3d_two(best_sim3, two_trajectories):
    best_r = best_sim3[0]
    best_t = best_sim3[1].reshape(3,1)
    best_c = best_sim3[2]
    pts_loc = two_trajectories[0:3, :]
    pts_map = two_trajectories[3:6, :]

    pts_scaled = pts_loc * best_c
    pts_scaled_transformed = np.matmul(best_r, pts_scaled) + best_t

    group_a = pts_scaled_transformed.transpose()
    group_b = pts_map.transpose()
    ax = plt.axes(projection="3d")
    ax.scatter(np.array(group_a)[:,0], np.array(group_a)[:,1],
                np.array(group_a)[:,2], c='b')
    ax.scatter(np.array(group_b)[:,0], np.array(group_b)[:,1],
                np.array(group_b)[:,2], c='r')
    ab_max = max(group_a.max(), group_b.max())
    ab_min = min(group_a.min(), group_b.min())
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([ab_min,ab_max])
    ax.set_ylim([ab_min,ab_max])
    ax.set_zlim([ab_min,ab_max])
    plt.show()
    return

def pt_global_to_camera(img, xyz):
    # img的pose是from world to image
    pt3d = np.matmul(img.qvec2rotmat(), xyz.reshape(3, 1)) + img.tvec.reshape(3, 1)
    if pt3d[2] < 1e-10: # 小于0是在相机后面, 接近0也不行, 都返回false
        return False, 0.0, 0.0
    return True, pt3d[0] / pt3d[2], pt3d[1] / pt3d[2]

def generate_id_pairs(images_loc, images_locmap):
    id_pair_list = []
    for key_locmap, img_locmap in images_locmap.items():
        for key_loc, img_loc in images_loc.items():
            if img_locmap.name == img_loc.name:
                id_pair_list.append([key_loc, key_locmap])
    return id_pair_list

def generate_pose_pairs(images_loc, images_locmap):
    '''
    loc做src, locmap做tgt, 为计算sim3做准备
    注意: images_locmap <= images_loc + images_map
    不是所有的loc图像都能成功注册到locmap
    需要寻找并维护images_locmap中loc图像在images_loc中的id对应关系
    '''
    # 寻找id对应关系
    id_pair_list = generate_id_pairs(images_loc, images_locmap)
    print('pair number = ', len(id_pair_list))
    if (len(id_pair_list) == 0):
        print('ERROR, pair number = 0')
        exit(0)
    
    positions_loc_locmap = np.zeros((6, len(id_pair_list)))

    for idx in range(len(id_pair_list)):
        key_loc = id_pair_list[idx][0]
        key_locmap = id_pair_list[idx][1]

        _, tvec = T_w2i_to_T_i2w(images_loc[key_loc])
        positions_loc_locmap[0, idx] = tvec[0]
        positions_loc_locmap[1, idx] = tvec[1]
        positions_loc_locmap[2, idx] = tvec[2]

        _, tvec = T_w2i_to_T_i2w(images_locmap[key_locmap])
        positions_loc_locmap[3, idx] = tvec[0]
        positions_loc_locmap[4, idx] = tvec[1]
        positions_loc_locmap[5, idx] = tvec[2]
    
    return id_pair_list, positions_loc_locmap

def sim3_transform_statistic_in_image(maybe_sim3, id_pair_list, 
                                    cameras_loc, images_loc, point3ds_loc, 
                                    images_locmap, threshold, inlier_number):
    maybe_r = maybe_sim3[0]
    maybe_t = maybe_sim3[1].reshape(3,1)
    maybe_c = maybe_sim3[2]
    mean_delta = 0.0
    count_all = 0
    inlier_id_pair_list = []
    # maybe_sim3是from loc to locmap
    for id_pair in id_pair_list: # 循环图像
        key_loc = id_pair[0]
        key_locmap = id_pair[1]
        # 取loc的image的所有东西
        img_loc = images_loc[key_loc]
        cam_loc = cameras_loc[img_loc.camera_id]
        # image数据结构里的xys和point3D_ids的长度是一致的
        count_one_img = 0
        for idx in range(len(img_loc.xys)): # 循环图像里的2d特征点
            pt3d_id = img_loc.point3D_ids[idx]
            if (pt3d_id < 0): # 2d观测没有对应的3d点
                continue
            pt3d_src = point3ds_loc[pt3d_id].xyz.reshape(3, 1)
            # 把loc全局坐标系下的点, sim3转换到locmap全局坐标系
            pt3d_src = pt3d_src * maybe_c # 尺度计算
            pt3d_tgt = np.matmul(maybe_r, pt3d_src) + maybe_t # 刚体计算
            # 图像用locmap全局坐标系下的pose, world2image
            ret, u, v = pt_global_to_camera(images_locmap[key_locmap], pt3d_tgt)
            if not ret: # 3d点没有落在相机前面, 跳过
                continue
            # 这里用loc的相机内参
            x, y = world2image_simple_radial(cam_loc.params, u, v) # 目前只支持simple_radial；更多支持，需要移植
            # 新pose, 新3d点, 旧cam, 得到新投影的2d点, 求残差
            delta_in_pixel = img_loc.xys[idx] - np.array([x, y])
            norm_delta = np.linalg.norm(delta_in_pixel)
            if norm_delta > threshold:
                continue
            mean_delta += norm_delta
            count_one_img += 1
            count_all += 1
        # 至少需要inlier_number个内点
        if count_one_img >= inlier_number: 
            inlier_id_pair_list.append(id_pair)
    if count_all == 0 or len(inlier_id_pair_list) == 0:
        return 1e10, inlier_id_pair_list # 错误结果，直接返回
    mean_delta = mean_delta / count_all
    return mean_delta, inlier_id_pair_list

def sim3_transform_statistic_in_trajectory(maybe_sim3, two_trajectories):
    # 没有阈值，所有点都是inlier，找误差最小的
    # 这里的误差没有尺度，不过都是在locmap坐标系下，量纲是一致的即可
    maybe_r = maybe_sim3[0]
    maybe_t = maybe_sim3[1].reshape(3,1)
    maybe_c = maybe_sim3[2]
    pts_loc = two_trajectories[0:3, :]
    pts_locmap = two_trajectories[3:6, :]
    pt3d_src = pts_loc * maybe_c
    pt3d_sim3 = np.matmul(maybe_r, pt3d_src) + maybe_t
    delta = pt3d_sim3 - pts_locmap
    sum_error = np.sum(np.linalg.norm(delta, axis=0))
    return sum_error

def seq_sfm_ransac_sim3(cameras_loc, images_loc, point3ds_loc, 
                        images_locmap, min_num_model, max_iter_num, threshold, inlier_number):
    # 生成用于ransac的元数据
    id_pair_list, two_trajectories = generate_pose_pairs(images_loc, images_locmap)
    # 初始化用于评估的量
    iterations = 0
    best_sim3 = None
    best_inlier_id_pair_list = []
    best_mean_delta = 1e10
    smallest_error = 1e10
    # ransac循环开始
    max_iter_num_a = len(id_pair_list) * max_iter_num
    while iterations < max_iter_num_a:
        # ------random------
        maybe_idxs, _ = random_partition(min_num_model, two_trajectories)
        maybe_inliers = two_trajectories[:, maybe_idxs]
        # ------fit   ------
        maybe_r, maybe_t, maybe_c = umeyama_alignment(maybe_inliers[0:3, : ], maybe_inliers[3:6, : ], True)
        maybe_sim3 = [maybe_r, maybe_t, maybe_c]
        # ------test  ------
        sum_error = sim3_transform_statistic_in_trajectory(maybe_sim3, two_trajectories)
        # ------compare------
        # 先用轨迹进行第一轮判断
        # 注意:这里的误差是无尺度的，但是都在locmap坐标系下，标准是一致的，可以比较大小
        # 用这个error的前提假设是所有的pose都是inlier，主要是为了提速，PNP校验很耗时
        if sum_error < smallest_error:
            # print('maybe smaller_error = ', sum_error)
            # 再用PNP进一步判断
            test_mean_delta, test_inlier_id_pair_list = \
                sim3_transform_statistic_in_image(maybe_sim3, id_pair_list, 
                                                    cameras_loc, images_loc, point3ds_loc,
                                                    images_locmap, threshold, inlier_number)
            # print('current mean_delta:', test_mean_delta, 'current inlier_number = ', len(test_inlier_id_pair_list))
            if len(test_inlier_id_pair_list) > len(best_inlier_id_pair_list):# 内点个数更多,直接更新最佳模型
                best_sim3 = maybe_sim3
                best_inlier_id_pair_list = test_inlier_id_pair_list
                best_mean_delta = test_mean_delta
                smallest_error = sum_error # 只有PNP校验成功的更小的轨迹error才是有效的error
                print('smaller_error = ', smallest_error)
                print('Update: more inliers -> best_mean_delta = ', best_mean_delta, 'best_inlier_number = ', len(best_inlier_id_pair_list))
            elif len(test_inlier_id_pair_list) == len(best_inlier_id_pair_list): # 内点个数相当,要进一步比较误差
                if (test_mean_delta < best_mean_delta):
                    best_sim3 = maybe_sim3
                    best_inlier_id_pair_list = test_inlier_id_pair_list
                    best_mean_delta = test_mean_delta
                    smallest_error = sum_error # 只有PNP校验成功的更小的轨迹error才是有效的error
                    print('smaller_error = ', smallest_error)
                    print('Update: less error -> best_mean_delta = ', best_mean_delta, 'best_inlier_number = ', len(best_inlier_id_pair_list))
        iterations += 1
    
    if best_sim3 is None:
        raise ValueError("did't meet fit acceptance criteria")
    
    # 调试用的可视化
    sim3_visualize_pts_3d_two(best_sim3, two_trajectories)
    return best_sim3, best_inlier_id_pair_list, best_mean_delta

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loc_model_path', required=True)
    parser.add_argument('--locmap_model_path', required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cameras_loc, images_loc, point3ds_loc = auto_read_model(args.loc_model_path)
    _, images_locmap, _ = auto_read_model(args.locmap_model_path)
    min_num_model = 4
    max_iter_num  = 1 # 这个参数会乘以图像数
    threshold = 12.0 # colmap默认的pixel-error-threshold是4，这里是它的3倍
    inlier_number = 5 # colmap默认的匹配成功的内点个数是15，这里是它的1/3, PNP至少需要4个点
    best_sim3, best_inlier_id_pair_list, best_mean_delta = \
        seq_sfm_ransac_sim3(cameras_loc, images_loc, point3ds_loc, 
                        images_locmap, min_num_model, max_iter_num, threshold, inlier_number)
    return

if __name__ == '__main__':
    main()