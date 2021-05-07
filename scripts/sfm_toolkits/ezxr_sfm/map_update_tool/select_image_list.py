# coding: utf-8
import os
import sys
import copy
import argparse
import numpy as np
import math
from shapely.geometry import Point, Polygon  # 多边形几何计算
sys.path.append("../")

from colmap_process.colmap_keyframe_selecter import auto_read_model
from colmap_process.create_file_list import write_image_list
from map_update_tool.label_image import read_position_constraint, read_degree_constraint, anticlockwise_degree_u2v
from python_scale_ezxr.transformations import quaternion_matrix

def select_images_with_constraints(model_path, polygon_txt_path, vector_txt_path):
    # 文件IO和检查
    print('auto_read_model...')
    _, images, _ = auto_read_model(model_path)
    print('read_position_constraint...')
    world_coordinate_a, polygon_pt_list = read_position_constraint(polygon_txt_path)
    is_with_vector = os.path.isfile(vector_txt_path)
    world_coordinate_b = None
    degree_constraint = None
    if (is_with_vector):
        print('read_degree_constraint...')
        world_coordinate_b, degree_constraint = read_degree_constraint(vector_txt_path)
        error_str = world_coordinate_a + ' vs ' + world_coordinate_b
        assert world_coordinate_a == world_coordinate_b, error_str
        error_str = world_coordinate_a + ' vs ' + 'XY or ZY or ZX'
        assert (world_coordinate_a == 'XY' or world_coordinate_a == 'ZY' or world_coordinate_a == 'ZX'), error_str    
    # 构建多边形
    polygon_pts = []
    for pt in polygon_pt_list:
        polygon_pts.append( (pt[0], pt[1]) )
    poly = Polygon(polygon_pts)
    # 构建局部坐标系下, 图像的朝向向量
    z_locl_axis = np.array([0.0, 0.0, 1.0])
    z_locl_axis.reshape(3, 1)
    valid_image_list = []
    print('select_images_with_constraints...')
    for _, img in images.items():
        # image里面的pose是W2C, 我们需要C2W
        rmat = quaternion_matrix(img.qvec)
        rmat = rmat[0:3, 0:3]
        rmat_W2C = rmat.T
        t_W2C = -rmat_W2C @ img.tvec
        # 根据不同的标注平面
        pos_row = None
        pos_col = None
        dir_row = None
        dir_col = None
        z_global_axis = rmat_W2C @ z_locl_axis
        if world_coordinate_a == 'XY':
            pos_row = t_W2C[0]
            pos_col = t_W2C[1]
            dir_row = z_global_axis[0]
            dir_col = z_global_axis[1]
        elif world_coordinate_a == 'ZY':
            pos_row = t_W2C[2]
            pos_col = t_W2C[1]
            dir_row = z_global_axis[2]
            dir_col = z_global_axis[1]
        elif world_coordinate_a == 'ZX':
            pos_row = t_W2C[2]
            pos_col = t_W2C[0]
            dir_row = z_global_axis[2]
            dir_col = z_global_axis[0]
        else:
            print('ERROR -> world_coordinate_a')
            exit(0)
        
        # 判断图像的position是否在多边形内
        pos_pt = Point(pos_row, pos_col)
        if not pos_pt.within(poly):
            continue
        if is_with_vector:
            # 判断图像的orientation是否在标注的角度区间内
            deg_cons = anticlockwise_degree_u2v(degree_constraint[0], degree_constraint[1], dir_row, dir_col)
            if deg_cons[2] > degree_constraint[2]:
                continue
        valid_image_list.append(img.name)
    valid_image_list = sorted(valid_image_list)
    return valid_image_list

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--polygon_txt_path', required=True)
    parser.add_argument('--vector_txt_path' , type=str, default='')
    parser.add_argument('--out_image_list_path' , required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    valid_image_list = select_images_with_constraints(args.model_path, args.polygon_txt_path, args.vector_txt_path)
    print('write_image_list...')
    write_image_list(args.out_image_list_path, valid_image_list)
    
if __name__ == "__main__":
    main()