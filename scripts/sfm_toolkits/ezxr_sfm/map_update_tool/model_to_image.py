# coding: utf-8
import os
import sys
import copy
import struct
import argparse
import numpy as np
import math
import cv2
import json
from plyfile import PlyData, PlyElement
import matplotlib.cm as cm
sys.path.append("../")

from trajectory_tools.colmap_to_tum_evo import read_geo
from python_scale_ezxr.transformations import quaternion_matrix

def value_to_color(color_map_rgb, color_map_resolution, min_value, max_value, cur_value):
    '''
    color_map_rgb是函数外生成的，只需要生成一次，可以用如下代码生成：
    color_map_resolution = 100
    color_map_x = np.linspace(0.0, 1.0, color_map_resolution+1)
    color_map_rgb = cm.get_cmap('jet')(color_map_x)[np.newaxis, :, :3]
    '''
    if cur_value < min_value:
        cur_value = min_value
    if cur_value > max_value:
        cur_value = max_value
    cur_value = (max_value - cur_value) / (max_value - min_value)
    cur_value = int(cur_value * color_map_resolution)
    cur_rgb = color_map_rgb[0][cur_value]
    cur_rgb = cur_rgb * 255
    cur_rgb_tuple = (int(cur_rgb[0]), int(cur_rgb[1]), int(cur_rgb[2]))
    return cur_rgb_tuple


def valid_grid_num_to_gray_image(XY_image, XY_RowCol, top_sup_percent):
    validnum = np.sum(XY_image > 0) # 非0元素的个数 
    validpercent =  100 - validnum * 1.0 / XY_image.size * top_sup_percent #压缩前1%的点云密集区域

    maxthr = np.percentile(XY_image,validpercent)
    print("validnum/totalgrid", validnum, "/", XY_image.size, " validpercent", validpercent, " maxthr", maxthr)

    XY_image[XY_image > maxthr] = maxthr # 统一上边界
    XY_image = XY_image / maxthr * 255.0 
    XY_image = 255 - XY_image.astype(np.uint8)

    # matrix变成彩色图
    tmp = np.zeros((XY_image.shape[0], XY_image.shape[1], 3), np.uint8)
    for idx in range(3):
        tmp[:,:,idx] = XY_image
    XY_image = tmp
    return XY_image

def converte_ply_to_image(ply_path, image_save_path, geos_path, resolution, top_sup_percent = 1):
    '''
    //      ___________________________________Y -> pixel_x(col)
    //      |             ^ X                 |
    //      |             |                   |
    //      |             |                   |
    //      |             |                   |
    //      | Y<-----------Z(out)             |
    //      |                                 |
    //      |                                 |
    //      |                                 |
    //      |                                 |
    //      |_________________________________|
    //      X -> pixel_y(row)

    //      ___________________________________X -> pixel_x(col)
    //      |             ^ Z                 |
    //      |             |                   |
    //      |             |                   |
    //      |             |                   |
    //      | X<-----------Y(out)             |
    //      |                                 |
    //      |                                 |
    //      |                                 |
    //      |                                 |
    //      |_________________________________|
    //      Z -> pixel_y(row)

    //      ___________________________________Y -> pixel_x(col)
    //      |             ^ Z                 |
    //      |             |                   |
    //      |             |                   |
    //      |             |                   |
    //      | Y<-----------X(in)              |
    //      |                                 |
    //      |                                 |
    //      |                                 |
    //      |                                 |
    //      |_________________________________|
    //      Z -> pixel_y(row)
    '''
    # 提前生成color map
    # 这个速度比较慢，且只需要运行一次
    color_map_resolution = 100
    color_map_x = np.linspace(0.0, 1.0, color_map_resolution+1)
    color_map_rgb = cm.get_cmap('jet')(color_map_x)[np.newaxis, :, :3]

    # ply数据结构转换成np.array
    plydata = PlyData.read(ply_path)
    X = plydata['vertex']['x']
    Y = plydata['vertex']['y']
    Z = plydata['vertex']['z']
    X_min = min(X)
    Y_min = min(Y)
    Z_min = min(Z)
    X_max = max(X)
    Y_max = max(Y)
    Z_max = max(Z)
    XY = np.zeros((len(plydata['vertex']['x']), 2))
    ZX = np.zeros((len(plydata['vertex']['x']), 2))
    ZY = np.zeros((len(plydata['vertex']['x']), 2))
    XY[:, 0] = np.array(X)
    XY[:, 1] = np.array(Y)

    ZX[:, 0] = np.array(Z)
    ZX[:, 1] = np.array(X)

    ZY[:, 0] = np.array(Z)
    ZY[:, 1] = np.array(Y)

    # 构建图像
    resolution_inv = 1.0 / resolution
    X_max_idx = int(math.ceil((X_max * resolution_inv)))
    Y_max_idx = int(math.ceil((Y_max * resolution_inv)))
    Z_max_idx = int(math.ceil((Z_max * resolution_inv)))

    X_min_idx = int(math.floor((X_min * resolution_inv)))
    Y_min_idx = int(math.floor((Y_min * resolution_inv)))
    Z_min_idx = int(math.floor((Z_min * resolution_inv)))

    X_delta_idx = X_max_idx - X_min_idx
    Y_delta_idx = Y_max_idx - Y_min_idx
    Z_delta_idx = Z_max_idx - Z_min_idx

    XY_image = np.zeros((X_delta_idx, Y_delta_idx))
    ZX_image = np.zeros((Z_delta_idx, X_delta_idx))
    ZY_image = np.zeros((Z_delta_idx, Y_delta_idx))
    # ZY_image = np.zeros((Z_delta_idx, Y_delta_idx, 3), np.uint8)

    # 统计每个网格落入3d点的个数
    XY_RowCol = [X_max_idx, Y_max_idx] - (np.ceil(XY * resolution_inv)).astype('int')
    for idx in range(XY_RowCol.shape[0]):
        XY_image[XY_RowCol[idx, 0], XY_RowCol[idx, 1]] += 1

    ZX_RowCol = [Z_max_idx, X_max_idx] - (np.ceil(ZX * resolution_inv)).astype('int')
    for idx in range(XY_RowCol.shape[0]):
        ZX_image[ZX_RowCol[idx, 0], ZX_RowCol[idx, 1]] += 1

    ZY_RowCol = [Z_max_idx, Y_max_idx] - (np.ceil(ZY * resolution_inv)).astype('int')
    for idx in range(XY_RowCol.shape[0]):
        ZY_image[ZY_RowCol[idx, 0], ZY_RowCol[idx, 1]] += 1

    XY_image = valid_grid_num_to_gray_image(XY_image, XY_RowCol, top_sup_percent)
    ZX_image = valid_grid_num_to_gray_image(ZX_image, ZX_RowCol, top_sup_percent)
    ZY_image = valid_grid_num_to_gray_image(ZY_image, ZY_RowCol, top_sup_percent)

    # 如果提供geos.txt, 画轨迹
    if os.path.isfile(geos_path):
        # 画geos.txt
        geo_dict = read_geo(geos_path)
        assert len(geo_dict) > 0, 'Error! empty geos!'
        geo_broken_match = None
        for key, value in geo_dict.items():
            if len(geo_dict[key]) == 4:
                geo_broken_match = np.zeros((len(geo_dict), 1))
            break
        geo_XY = np.zeros((len(geo_dict), 2))
        geo_ZY = np.zeros((len(geo_dict), 2))
        geo_ZX = np.zeros((len(geo_dict), 2))

        idx = 0
        for _, value in geo_dict.items():
            geo_XY[idx, 0] = value[0]
            geo_XY[idx, 1] = value[1]

            geo_ZX[idx, 0] = value[2]
            geo_ZX[idx, 1] = value[0]

            geo_ZY[idx, 0] = value[2]
            geo_ZY[idx, 1] = value[1]

            if geo_broken_match is not None:
                geo_broken_match[idx] = value[3]

            idx += 1
        
        geo_XY_traj = [X_max_idx, Y_max_idx] - (np.ceil(geo_XY * resolution_inv)).astype('int')
        geo_ZX_traj = [Z_max_idx, X_max_idx] - (np.ceil(geo_ZX * resolution_inv)).astype('int')
        geo_ZY_traj = [Z_max_idx, Y_max_idx] - (np.ceil(geo_ZY * resolution_inv)).astype('int')
        if geo_broken_match is not None:
            for idx in range(geo_XY.shape[0]):
                cur_rgb_tuple = value_to_color(color_map_rgb, color_map_resolution, 0, 50, geo_broken_match[idx])
                XY_image = cv2.circle(XY_image, (geo_XY_traj[idx, 1], geo_XY_traj[idx, 0]), 3, cur_rgb_tuple, 1)
                ZX_image = cv2.circle(ZX_image, (geo_ZX_traj[idx, 1], geo_ZX_traj[idx, 0]), 3, cur_rgb_tuple, 1)
                ZY_image = cv2.circle(ZY_image, (geo_ZY_traj[idx, 1], geo_ZY_traj[idx, 0]), 3, cur_rgb_tuple, 1)
        else:
            for idx in range(geo_XY.shape[0]):
                XY_image = cv2.circle(XY_image, (geo_XY_traj[idx, 1], geo_XY_traj[idx, 0]), 3, (0, 0, 255), 1)
                ZX_image = cv2.circle(ZX_image, (geo_ZX_traj[idx, 1], geo_ZX_traj[idx, 0]), 3, (0, 0, 255), 1)
                ZY_image = cv2.circle(ZY_image, (geo_ZY_traj[idx, 1], geo_ZY_traj[idx, 0]), 3, (0, 0, 255), 1)
    
    # 画右手坐标系
    draw_axis = True
    if (draw_axis):
        length_axis = 5.0
        axis_idx = int(math.ceil(length_axis * resolution_inv))

        XY_origin = [X_max_idx, Y_max_idx]
        XY_x_axis = [X_max_idx - axis_idx, Y_max_idx]
        XY_y_axis = [X_max_idx, Y_max_idx - axis_idx]
        XY_image = cv2.arrowedLine(XY_image, (XY_origin[1], XY_origin[0]), (XY_x_axis[1], XY_x_axis[0]), (0, 0, 255), 2) 
        XY_image = cv2.arrowedLine(XY_image, (XY_origin[1], XY_origin[0]), (XY_y_axis[1], XY_y_axis[0]), (0, 255, 0), 2)

        ZX_origin = [Z_max_idx, X_max_idx]
        ZX_z_axis = [Z_max_idx - axis_idx, X_max_idx]
        ZX_x_axis = [Z_max_idx, X_max_idx - axis_idx]
        ZX_image = cv2.arrowedLine(ZX_image, (ZX_origin[1], ZX_origin[0]), (ZX_z_axis[1], ZX_z_axis[0]), (255, 0, 0), 2) 
        ZX_image = cv2.arrowedLine(ZX_image, (ZX_origin[1], ZX_origin[0]), (ZX_x_axis[1], ZX_x_axis[0]), (0, 0, 255), 2)

        ZY_origin = [Z_max_idx, Y_max_idx]
        ZY_z_axis = [Z_max_idx - axis_idx, Y_max_idx]
        ZY_y_axis = [Z_max_idx, Y_max_idx - axis_idx]
        ZY_image = cv2.arrowedLine(ZY_image, (ZY_origin[1], ZY_origin[0]), (ZY_z_axis[1], ZY_z_axis[0]), (255, 0, 0), 2) 
        ZY_image = cv2.arrowedLine(ZY_image, (ZY_origin[1], ZY_origin[0]), (ZY_y_axis[1], ZY_y_axis[0]), (0, 255, 0), 2)
    
    # 保存图像
    XY_image_save_path = image_save_path[0:-4] + '_XY.png'
    ZX_image_save_path = image_save_path[0:-4] + '_ZX.png'
    ZY_image_save_path = image_save_path[0:-4] + '_ZY.png'

    cv2.imwrite(XY_image_save_path, XY_image)
    cv2.imwrite(ZX_image_save_path, ZX_image)
    cv2.imwrite(ZY_image_save_path, ZY_image)

    # 保存配置文件, 2d->3d的转换方法
    XY_txt_save_path = image_save_path[0:-4] + '_XY.txt'
    ZX_txt_save_path = image_save_path[0:-4] + '_ZX.txt'
    ZY_txt_save_path = image_save_path[0:-4] + '_ZY.txt'

    XY_txt = open(XY_txt_save_path, 'w')
    XY_txt.write('# XY_TRANSFORM->imagematrix_to_world:\n')
    XY_txt.write('# X = resolution * (max_X_ROW - image_row)\n')
    XY_txt.write('# Y = resolution * (max_Y_COL - image_col)\n')
    XY_txt.write('resolution(m):' + str(resolution) + '\n')
    XY_txt.write('max_X_ROW:' + str(X_max_idx) + '\n')
    XY_txt.write('max_Y_COL:' + str(Y_max_idx) + '\n')
    XY_txt.close()

    # 适配mapLT需要的json
    XY_json_save_path = image_save_path[0:-4] + '_XY.json'
    XY_json_data = {}
    XY_json_data['X'] = 'resolution * (max_X_ROW - image_row)'
    XY_json_data['Y'] = 'resolution * (max_Y_COL - image_col)'
    XY_json_data['unit'] = 'm'
    XY_json_data['resolution'] = str(resolution)
    XY_json_data['max_X_ROW'] = str(X_max_idx)
    XY_json_data['max_Y_COL'] = str(Y_max_idx)
    XY_json_data['T_visual_to_xx'] = np.identity(4, dtype=float).tolist()
    json_file = open(XY_json_save_path, 'w')
    json.dump(XY_json_data, json_file, indent=4)

    ZX_txt = open(ZX_txt_save_path, 'w')
    ZX_txt.write('# ZX_TRANSFORM->imagematrix_to_world:\n')
    ZX_txt.write('# Z = resolution * (max_Z_ROW - image_row)\n')
    ZX_txt.write('# X = resolution * (max_X_COL - image_col)\n')
    ZX_txt.write('resolution(m):' + str(resolution) + '\n')
    ZX_txt.write('max_Z_ROW:' + str(Z_max_idx) + '\n')
    ZX_txt.write('max_X_COL:' + str(X_max_idx) + '\n')
    ZX_txt.close()

    ZY_txt = open(ZY_txt_save_path, 'w')
    ZY_txt.write('# ZY_TRANSFORM->imagematrix_to_world:\n')
    ZY_txt.write('# Z = resolution * (max_Z_ROW - image_row)\n')
    ZY_txt.write('# Y = resolution * (max_Y_COL - image_col)\n')
    ZY_txt.write('resolution(m):' + str(resolution) + '\n')
    ZY_txt.write('max_Z_ROW:' + str(Z_max_idx) + '\n')
    ZY_txt.write('max_Y_COL:' + str(Y_max_idx) + '\n')
    ZY_txt.close()
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply_path', required=True)
    parser.add_argument('--geos_path', default='', type=str)
    parser.add_argument('--image_save_path' , required=True)
    parser.add_argument('--resolution', default=0.10, type=float)
    parser.add_argument('--top_sup_percent', default=1, type=int)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    converte_ply_to_image(args.ply_path, args.image_save_path, args.geos_path, args.resolution, args.top_sup_percent)
    
if __name__ == "__main__":
    main()