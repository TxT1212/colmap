# coding: utf-8
import os
import sys
import math
import numpy as np
import argparse
from scipy import stats  # 统计
from matplotlib import pyplot as plt
from shapely.geometry import Polygon  # 多边形几何计算
import re # 正则表达式

import networkx as nx # 图, 相关算法

sys.path.append('../')
from colmap_process.colmap_db_parser import *
from colmap_process.colmap_read_write_model import *
from python_scale_ezxr.transformations import quaternion_matrix, quaternion_from_matrix
from colmap_process.create_file_list import write_image_list
# -----------------------文件IO start---------------------------
def auto_read_model(model_folder):
    '''
    # Camera list with one line of data per camera:
    #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    # Number of cameras: 3

    # Image list with two lines of data per image:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)

    # 3D point list with one line of data per point:
    #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    # Number of points: 3, mean track length: 3.3334
    '''
    cameras_path_name = model_folder + '/cameras.txt'
    images_path_name = model_folder + '/images.txt'
    points3d_path_name = model_folder + '/points3D.txt'
    if os.path.exists(cameras_path_name) and \
        os.path.exists(images_path_name) and \
        os.path.exists(points3d_path_name):
        cameras, images, points3d = read_model(model_folder, '.txt')
        return cameras, images, points3d
    cameras_path_name = model_folder + '/cameras.bin'
    images_path_name = model_folder + '/images.bin'
    points3d_path_name = model_folder + '/points3D.bin'
    if os.path.exists(cameras_path_name) and \
        os.path.exists(images_path_name) and \
        os.path.exists(points3d_path_name):
        cameras, images, points3d = read_model(model_folder, '.bin')
        return cameras, images, points3d
    raise ValueError("Failed to read model, please check the model folder!")

# -----------------------文件IO end---------------------------

# -----------------------多视图几何 相关计算 start---------------------------
def quantify_relative_pose(rmat, tvec):
    pose = np.eye(4)
    pose[0:3, 0:3] = rmat
    qwxyz = quaternion_from_matrix(pose)
    # 根据四元数求解角度, 单位是:度
    relative_degree = np.arccos( np.abs(qwxyz[0]) ) * 57.295779513
    # 相对距离是没有尺度没有单位的
    relative_distance = np.linalg.norm(tvec)
    return relative_degree, relative_distance

# colmap model里读取出来的img的pose是T_w2i, 即world to image, 但轨迹是image to world
def T_w2i_to_T_i2w(img):
    r = quaternion_matrix(img.qvec)[0:3, 0:3]
    rmat = r.transpose()
    tvec = img.tvec
    tnew = -rmat @ tvec
    return rmat, tnew

# 两个轨迹都是image to world, 才能计算相对位姿
def cal_relative_pose(rmat0, tvec0, rmat1, tvec1):
    # inverse matrix
    rmat0_inv = rmat0.transpose()
    tvec0_inv = -rmat0_inv @ tvec0

    pose0_inv = np.eye(4)
    pose0_inv[0:3, 0:3] = rmat0_inv
    pose0_inv[0:3, 3] = tvec0_inv

    pose1 = np.eye(4)
    pose1[0:3, 0:3] = rmat1
    pose1[0:3, 3] = tvec1

    T_1_to_0 = pose0_inv @ pose1
    return T_1_to_0[0:3, 0:3], T_1_to_0[0:3, 3]

def image2world(params, x, y):
    fx = 0
    fy = 0
    cx = 0
    cy = 0
    if len(params) == 3:
        fx = params[0]
        fy = fx
        cx = params[1]
        cy = params[2]
    elif len(params) >= 4:
        fx = params[0]
        fy = params[1]
        cx = params[2]
        cy = params[3]

    u = (x - cx) / fx
    v = (y - cy) / fy
    return u, v

def world2image(params, u, v):
    fx = 0
    fy = 0
    cx = 0
    cy = 0
    if len(params) == 3:
        fx = params[0]
        fy = fx
        cx = params[1]
        cy = params[2]
    elif len(params) >= 4:
        fx = params[0]
        fy = params[1]
        cx = params[2]
        cy = params[3]

    x = fx * u + cx
    y = fy * v + cy
    return x, y
# -----------------------多视图几何 相关计算 end---------------------------

def pencent_str(a):
    return str(format(a * 100, '.2f')) + '%'

def img_node_id_str(img_id):
    return 'img_' + str(img_id)

def pt3d_node_id_str(pt3d_id):
    return 'pt3d_' + str(pt3d_id)

# -----------------------图 相关计算 start---------------------------
def statistic_of_node_degree(graph_of_sfm, img_nodes):
    '''
    统计节点的度, 这里只统计img节点, 能看到的3d点
    注意, 这里的graph假设已经剔除了invalid的3d点
    '''
    img_nodes_count_inv = 1.0 / len(img_nodes)
    count_threshold = [25, 50, 100, 200, 400, 800, 1600]
    img_node_degree_hist = np.zeros(len(count_threshold))
    for img_node in img_nodes:
        img_node_degree = graph_of_sfm.degree(img_node)
        for idx in range(len(count_threshold)):
            if img_node_degree > count_threshold[idx]:
                img_node_degree_hist[idx] = img_node_degree_hist[idx] + 1
    for idx in range(len(count_threshold)):
        print('≥', count_threshold[idx], ':', img_node_degree_hist[idx], pencent_str(img_node_degree_hist[idx] * img_nodes_count_inv))
    return

def calculate_intersected_area(cam, img0, img1):
    '''
    把相机0的矩形四个点, 反投影到单位平面上, 再根据相机0和相机1的相对位姿, 投影到相机1的相平面上
    然后用多边形(这里是两个四边形)相交面积, 求解相交面积占原图像面积的百分比
    '''
    # 使用单位平面, 针孔模型, 计算两个四边形的intersection
    xy_bb = [(0.0, 0.0), (cam.width, 0.0), (cam.width, cam.height), (0.0, cam.height)]
    # 把第一个图像的矩形四个点,反投影到单位平面上
    uv_bb = np.ones((3, 4))
    for idx in range(len(xy_bb)):
        uv_bb[0, idx], uv_bb[1, idx] = image2world(cam.params, xy_bb[idx][0], xy_bb[idx][1])
    # 计算相对位姿
    rmat0, tvec0 = T_w2i_to_T_i2w(img0)
    rmat1, tvec1 = T_w2i_to_T_i2w(img1)
    rmat, tvec = cal_relative_pose(rmat0, tvec0, rmat1, tvec1)
    #计算单位平面上的四个点投影到第二个图像
    xy_bb_new = xy_bb.copy() # 这里必须用copy,不然会内存共享
    for idx in range(len(xy_bb)):
        pt_uv = np.matmul(rmat, uv_bb[:, idx].reshape(3, 1)) + tvec.reshape(3, 1)
        pt_uv = pt_uv / pt_uv[2]
        x, y = world2image(cam.params, pt_uv[0], pt_uv[1])
        xy_bb_new[idx] = (x[0], y[0])
    # 使用多边形库, 求解两个四边形的相交面积, 面积的单位是像素^2, 我们只关心百分比
    p = Polygon(xy_bb)
    q = Polygon(xy_bb_new)
    if not p.is_valid or not q.is_valid:
        return 0.0
    iter_ratio = p.intersection(q).area * 1.0 / (cam.width * cam.height)
    return iter_ratio

def calculate_common_pt3d_ratio(img0, img1):
    pt3d0_vec = set()
    pt3d1_vec = set()
    for pt3d_id in img0.point3D_ids:
        if pt3d_id < 0:  # 无效id
            continue
        pt3d0_vec.add(pt3d_id)
    for pt3d_id in img1.point3D_ids:
        if pt3d_id < 0:  # 无效id
            continue
        pt3d1_vec.add(pt3d_id)
    common_pt3d_vec = pt3d0_vec.intersection(pt3d1_vec)
    return len(common_pt3d_vec) * 1.0 / len(pt3d0_vec), len(common_pt3d_vec)

def construct_sfm_graph(cameras, images, points3d):
    '''
    根据colmap model构建graph
    注意1: 后续所有的subgraph都是graph的子图, 无需构建, 只需要通过nodes的子集构建即可
    注意2: 构建subgraph的时候, 需要对pt3d的节点做invalid filter
    '''
    graph_of_sfm = nx.Graph()
    print('---> construct_sfm_graph...')
    # 因为networkx的graph的node都是一个类别, 我们通过node的key的前缀区分该node是img还是pt3d
    for img_id, img in images.items():
        if img_id < 0: # 无效id
            continue
        img_id_str = img_node_id_str(img_id)
        graph_of_sfm.add_node(img_id_str)
        for pt3d_id in img.point3D_ids:
            if pt3d_id < 0:  # 无效id
                continue
            pt3d_id_str = pt3d_node_id_str(pt3d_id)
            # 这里可以直接添加node, 因为同样的node id只会添加一个, 不用担心重复
            graph_of_sfm.add_node(pt3d_id_str)
            graph_of_sfm.add_edge(img_id_str, pt3d_id_str)
    # 预先把node区分提取
    all_nodes = list(graph_of_sfm.nodes())
    img_nodes_graph = list(filter(lambda x: re.match('img_', x) != None, all_nodes))
    pt3d_nodes_graph = list(filter(lambda x: re.match('pt3d_', x) != None, all_nodes))
    print('graph info: img_num = ', len(img_nodes_graph), \
        ', pt3d_num = ', len(pt3d_nodes_graph), \
        ', average_img_observe = ', 1.0 * len(pt3d_nodes_graph) / len(img_nodes_graph))
    # statistic_of_node_degree(graph_of_sfm, img_nodes_graph)
    return graph_of_sfm, img_nodes_graph, pt3d_nodes_graph

def resort_images_by_name(images):
    '''
    colmap生成database的时候,可能不按照文件名的顺序排序，这里重排一下
    '''  
    return dict(sorted(images.items(), key=lambda item: item[1].name))

def select_kf_by_img_intersection(cameras, images, points3d, \
                            fat_threshold = 0.9, thin_threshold = 0.75, \
                            pt_ratio_threshold = 0.25, pt_num_threshold = 30):
    '''
    根据相交面积的高低阈值(thin fat), 动态地选择sub img nodes
    注意: 存在少量两帧相交面积 > fat 或者 相交面积 < thin, 
    说明: 如果colmap model里的两帧图像本身相交面积就很小, 它就是 < thin
         如果本身 > fat, 但是一帧帧往后去, 依然 > fat, 再往后取一帧又 < thin, 我们更愿意它 > fat
         pt_ratio_threshold是根据下一帧的3d点在当前帧的可观测百分比, 做兜底
         pt_num_threshold是根据下一帧的3d点在当前帧的可观测数量,做兜底
         面积比例做空间采样
         特征比例做几何兜底     
    '''
    if (fat_threshold >= 1.0 or thin_threshold >= fat_threshold or thin_threshold <= 0):
        raise ValueError("Wrong ---> fat_threshold or thin_threshold")
    if (pt_ratio_threshold >= 1.0 or  pt_ratio_threshold <= 0):
        raise ValueError("Wrong ---> pt_ratio_threshold")
    if (pt_num_threshold <= 0):
        raise ValueError("Wrong ---> pt_num_threshold")
    
    images_sorted = resort_images_by_name(images)
    imgs = []
    for _, img in images_sorted.items():
        imgs.append(img)
    # 根据高低阈值, 动态地选取sub img nodes
    idx = 0
    sub_img_nodes = []
    sub_img_nodes.append(img_node_id_str(imgs[idx].id)) # 添加第一帧
    while (idx + 1 < len(imgs)): # 进入该循环, 至少需要两个图像, 所以判断idx + 1
        img0 = imgs[idx]
        next_idx = idx + 1
        img1 = imgs[next_idx]
        # 初始两帧判断
        area_ratio = calculate_intersected_area(cameras[img0.camera_id], img0, img1)
        pt_ratio, pt0_num = calculate_common_pt3d_ratio(img0, img1)
        # print('cur area_ratio = ', area_ratio, ', pt_ratio = ', pt_ratio, 'pt0_num = ', pt0_num)
        if pt_ratio <= pt_ratio_threshold or pt0_num <= pt_num_threshold: # 特征兜底, 如果太小, 直接返回
            # print('too small pt')
            sub_img_nodes.append(img_node_id_str(imgs[next_idx].id)) # 添加节点
            idx = next_idx # 更新起点
            continue
        if area_ratio > fat_threshold: # 如果太大, 就继续往下找
            # print('too fat')            
            # 特征比例还行, 继续往下找
            next_idx = next_idx + 1
            while(next_idx < len(imgs)):
                img1 = imgs[next_idx]
                area_ratio = calculate_intersected_area(cameras[img0.camera_id], img0, img1)
                pt_ratio, pt0_num = calculate_common_pt3d_ratio(img0, img1)
                if pt_ratio <= pt_ratio_threshold or pt0_num <= pt_num_threshold: # 特征兜底, 如果太小, 直接返回
                    # print('too fat->too small pt')
                    next_idx = next_idx - 1 # 修复bug, 退回上一帧
                    break
                if area_ratio > fat_threshold: # 如果太大, 就继续往下找
                    # print('too fat->too fat, ', area_ratio, ', ', pt_ratio)
                    next_idx = next_idx + 1
                    continue
                elif area_ratio <= fat_threshold and area_ratio >= thin_threshold: # 较为合理的overlap
                    # print('too fat->ok , ', area_ratio, ', ', pt_ratio)
                    break
                else: # 已经面积交叉太少了, 退回上一个, 宁愿overlap较大, 也不能较小
                    # print('too fat->too thin, ', area_ratio, ', ', pt_ratio)
                    next_idx = next_idx - 1 # 回退到前一帧
                    break
        elif area_ratio <= fat_threshold and area_ratio >= thin_threshold: # 较为合理的overlap
            # print('ok')
            pass
        else: # 如果一上来就太小, 那我也没办法
            # print('warnning! The overlap of two adjacent frames < ', thin_threshold)
            pass
        if (next_idx < len(imgs)):
            sub_img_nodes.append(img_node_id_str(imgs[next_idx].id))
        idx = next_idx # 更新下一帧的起点
    return sub_img_nodes

def statistics_adaptive_threshold_in_seq_model(imgs, pt_ratio_threshold, pt_num_threshold):
    min_pt_ratio = 1.0
    min_pt0_num = 1000000
    for idx in range(0, len(imgs) - 1):
        img0 = imgs[idx]
        next_idx = idx + 1
        img1 = imgs[next_idx]
        # 初始两帧判断
        pt_ratio, pt0_num = calculate_common_pt3d_ratio(img0, img1)
        # print(img0.name, img1.name)
        # print('cur pt_ratio = ', pt_ratio, ', pt0_num = ', pt0_num)
        if pt_ratio > pt_ratio_threshold and min_pt_ratio > pt_ratio:
            min_pt_ratio = pt_ratio
        if pt0_num > pt_num_threshold and min_pt0_num > pt0_num:
            min_pt0_num = pt0_num
    return min_pt_ratio, min_pt0_num

def select_kf_by_img_pt_ratio(cameras, images, points3d, \
                            pt_ratio_threshold = 0.25, pt_num_threshold = 30, use_adaptive_threshold = False):
    if (pt_ratio_threshold >= 1.0 or  pt_ratio_threshold <= 0):
        raise ValueError("Wrong ---> pt_ratio_threshold")
    if (pt_num_threshold <= 0):
        raise ValueError("Wrong ---> pt_num_threshold")
    imgs = []
    # model里是按顺序排的,但是直接用values不一定按顺序排,这里就笨一点
    images_sorted = resort_images_by_name(images)
    imgs = []
    for _, img in images_sorted.items():
        imgs.append(img)
    if use_adaptive_threshold:
        pt_ratio_threshold, pt_num_threshold = statistics_adaptive_threshold_in_seq_model(imgs, pt_ratio_threshold, pt_num_threshold)
        print('weak_threshold: pt_ratio_threshold = ', pt_ratio_threshold, ', pt_num_threshold = ', pt_num_threshold)
    # 统计哪些帧是被pt_num_threshold保留的，哪些帧是被pt_ratio_threshold保留的
    count_pt_num = 0
    # 根据阈值, 动态地选取sub img nodes
    idx = 0
    sub_img_nodes = []
    sub_img_nodes.append(img_node_id_str(imgs[idx].id)) # 添加第一帧
    while (idx + 1 < len(imgs)): # 进入该循环, 至少需要两个图像, 所以判断idx + 1
        img0 = imgs[idx]
        next_idx = idx + 1
        img1 = imgs[next_idx]
        # 初始两帧判断
        pt_ratio, pt0_num = calculate_common_pt3d_ratio(img0, img1)
        # print('pt_ratio = ', pt_ratio, 'pt0_num = ', pt0_num)
        if pt0_num <= pt_num_threshold: # 特征兜底, 如果太小, 直接返回
            # print('too small pt')
            sub_img_nodes.append(img_node_id_str(imgs[next_idx].id)) # 添加节点
            idx = next_idx # 更新起点
            count_pt_num += 1
            continue
        if pt_ratio > pt_ratio_threshold: # 如果太大, 就继续往下找         
            # 特征比例还行, 继续往下找
            next_idx = next_idx + 1
            while(next_idx < len(imgs)):
                img1 = imgs[next_idx]
                pt_ratio, pt0_num = calculate_common_pt3d_ratio(img0, img1)
                if pt0_num <= pt_num_threshold: # 特征兜底, 如果太小, 直接返回
                    # print('too small pt')
                    next_idx = next_idx - 1 # 回退到前一帧
                    count_pt_num += 1
                    break
                if pt_ratio > pt_ratio_threshold: # 如果太大, 就继续往下找
                    # print('too fat->too fat, ', area_ratio, ', ', pt_ratio)
                    next_idx = next_idx + 1
                    continue
                else: # 已经面积交叉太少了, 退回上一个, 宁愿overlap较大, 也不能较小
                    # print('too fat->too thin, ', area_ratio, ', ', pt_ratio)
                    next_idx = next_idx - 1 # 回退到前一帧
                    break
        else: # 如果一上来就太小, 那我也没办法
            # print('warnning! The overlap of two adjacent frames < ', thin_threshold)
            pass
        if (next_idx < len(imgs)):
            sub_img_nodes.append(img_node_id_str(imgs[next_idx].id))
        idx = next_idx # 更新下一帧的起点
    return sub_img_nodes, count_pt_num

def uniformly_downsample_img_nodes(img_nodes, faction_num):
    sub_img_nodes = {}
    for idx in range(faction_num):
        sub_img_nodes[idx] = []
    count = 0
    for node in img_nodes:
        num = count % faction_num
        sub_img_nodes[num].append(node)
        count = count + 1
    return sub_img_nodes

def get_subgraph_with_valid_pt3ds(graph_of_sfm, sub_img_nodes, pt3d_nodes):
    # 用部分img node和全部pt3d node构建subgrpah
    sub_nodes0 = sub_img_nodes.copy()
    sub_nodes0.extend(pt3d_nodes)
    subgraph0 = graph_of_sfm.subgraph(sub_nodes0)
    valid_pt3ds = []
    for node in sub_nodes0:
        if node.startswith('pt3d_'): # 判断是3d点的node
            if subgraph0.degree(node) >= 3: # 如果该3d点, 不能被3个及以上img的node看到, 记为不合理3d点
                valid_pt3ds.append(node)
    
    print('subgraph info: img_num = ', len(sub_img_nodes), \
        ', pt3d_num = ', len(valid_pt3ds), \
        ', average_img_observe = ', 1.0 * len(valid_pt3ds) / len(sub_img_nodes))
    
    sub_nodes1 = sub_img_nodes.copy()
    sub_nodes1.extend(valid_pt3ds)
    subgraph1 = graph_of_sfm.subgraph(sub_nodes1)
    return subgraph1
# -----------------------图 相关计算 end---------------------------

def generate_kfs_list(output_kf_list_path, images, sub_img_nodes):
    print('--->generate_kfs_list...')
    kfs_list = []
    for node in sub_img_nodes:
        # graph里的img的id是img_xxx,所以把前缀去掉再转换成int
        img_id = int(node[4:])
        name = images[img_id].name
        kfs_list.append(name)
    kfs_list = sorted(kfs_list)
    write_image_list(output_kf_list_path, kfs_list)
    return

def select_kfs(small_model_folder, small_kf_list_path, fat_threshold, thin_threshold, pt_ratio_threshold, pt_num_threshold, use_adaptive_threshold = False):
    # 读colmap model
    cameras, images, points3d = auto_read_model(small_model_folder)
    # 构建graph
    graph_of_sfm, img_nodes, pt3d_nodes = construct_sfm_graph(cameras, images, points3d)
    # 视野均匀筛选关键帧: 根据视野相交面积,挑选关键帧
    # sub_img_nodes = select_kf_by_img_intersection(cameras, images, points3d, 
    #                                             fat_threshold, thin_threshold, 
    #                                             pt_ratio_threshold, pt_num_threshold)
    # 只根据3d点比例筛选关键帧，3d点个数做兜底
    sub_img_nodes, count_pt_num = select_kf_by_img_pt_ratio(cameras, images, points3d, 
                                                pt_ratio_threshold, pt_num_threshold, use_adaptive_threshold)
    # 构建关键帧组成的subgraph, 该subgraph是后续算法的输入
    kf_subgraph = get_subgraph_with_valid_pt3ds(graph_of_sfm, sub_img_nodes, pt3d_nodes)
    # 统计关键帧组成的subgraph, 图像观测3d点的情况
    # statistic_of_node_degree(kf_subgraph, sub_img_nodes)
    generate_kfs_list(small_kf_list_path, images, sub_img_nodes)
    # 返回图像数量信息
    kf_img_count = len(sub_img_nodes)
    image_count = len(img_nodes)
    ratio = float(kf_img_count) / float(image_count)
    print('kf_img_count/image_count = ', kf_img_count, '/', image_count, ' = ', ratio)
    return kf_img_count, image_count, count_pt_num

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_model_folder', required=True)
    parser.add_argument('--small_kf_list_path', required=True)
    parser.add_argument('--fat_threshold', type=float, default=0.76)
    parser.add_argument('--thin_threshold', type=float, default=0.74)
    parser.add_argument('--pt_ratio_threshold', type=float, default=0.25)
    parser.add_argument('--pt_num_threshold', type=int, default=30)
    parser.add_argument('--use_adaptive_threshold', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    select_kfs(args.small_model_folder, args.small_kf_list_path, 
            args.fat_threshold, args.thin_threshold, 
            args.pt_ratio_threshold, args.pt_num_threshold,
            args.use_adaptive_threshold)
    return

if __name__ == '__main__':
    main()