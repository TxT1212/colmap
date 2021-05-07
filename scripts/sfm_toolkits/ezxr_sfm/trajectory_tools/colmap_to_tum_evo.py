# coding: utf-8
import os
import sys
import struct
import argparse
import numpy as np

sys.path.append("../")

from colmap_process.colmap_read_write_model import *
from colmap_process.colmap_export_geo import *
from colmap_process.colmap_keyframe_selecter import auto_read_model
from python_scale_ezxr.transformations import quaternion_matrix
def read_geo(file_path_name):
    fo = open(file_path_name, "r")
    geo_dict = {}
    for line in fo.readlines(): # 依次读取每行  
        line = line.strip() # 去掉每行头尾空白  
        elements = line.split(' ')
        if len(elements) == 4: # 标准geos
            geo_dict[elements[0]] = [float(elements[1]), float(elements[2]), float(elements[3])]
        elif len(elements) == 5: # 带有broken match数量的geos
            geo_dict[elements[0]] = [float(elements[1]), float(elements[2]), float(elements[3]), float(elements[4])]
        else:
            assert False, 'geos file has wrong format!'
    fo.close()
    return geo_dict

def write_common_name_dict(src_output_path, tgt_output_path, common_name_dict):
    # 根据文件名排序
    common_name_dict = dict(sorted(common_name_dict.items(), key=lambda item: item[0]))
    # 写文件 
    src_file = open(src_output_path,'w')
    tgt_file = open(tgt_output_path,'w')
    idx = 0
    for key, value in common_name_dict.items():
        src_line = str(idx) + ' ' + str(value[0][0]) + ' ' + str(value[0][1]) + ' ' + str(value[0][2]) + ' 0 0 0 1\n'
        tgt_line = str(idx) + ' ' + str(value[1][0]) + ' ' + str(value[1][1]) + ' ' + str(value[1][2]) + ' 0 0 0 1\n'
        src_file.write(src_line)
        tgt_file.write(tgt_line)
        idx += 1
    src_file.close()
    tgt_file.close()
    return

def get_image_position(img):
    rmat = quaternion_matrix(img.qvec)
    r = rmat[0:3, 0:3]
    r_inv = r.T
    tvec = img.tvec
    tnew = -np.matmul(r_inv, tvec)
    return tnew

def geo_and_images_to_tum_traj(src_output_path, tgt_output_path, geo_dict, images):
    '''
    geo是src
    images是tgt
    '''
    # 寻找相同name的变量，只存储xyz
    common_name_dict = {}
    for key, value in images.items():
        if value.name in geo_dict:
            geo_xyz = geo_dict[value.name]
            img_xyz = get_image_position(value)
            common_name_dict[value.name] = [geo_xyz, img_xyz]
    print('common image number = ', len(common_name_dict))
    write_common_name_dict(src_output_path, tgt_output_path, common_name_dict)
    return

def images_and_geo_to_tum_traj(src_output_path, tgt_output_path, images, geo_dict):
    '''
    images是src
    geo是tgt
    '''
    # 寻找相同name的变量，只存储xyz
    common_name_dict = {}
    for key, value in images.items():
        if value.name in geo_dict:
            geo_xyz = geo_dict[value.name]
            img_xyz = get_image_position(value)
            common_name_dict[value.name] = [img_xyz, geo_xyz]
    print('common image number = ', len(common_name_dict))
    write_common_name_dict(src_output_path, tgt_output_path, common_name_dict)
    return
    

def geo_and_geo_to_tum_traj(src_output_path, tgt_output_path, src_geo_dict, tgt_geo_dict):
    # 寻找相同name的变量，只存储xyz
    common_name_dict = {}
    for key, value in src_geo_dict.items():
        if key in tgt_geo_dict:
            src_xyz = src_geo_dict[key]
            tgt_xyz = tgt_geo_dict[key]
            common_name_dict[key] = [src_xyz, tgt_xyz]
    print('common image number = ', len(common_name_dict))
    write_common_name_dict(src_output_path, tgt_output_path, common_name_dict)
    return

def images_and_images_to_tum_traj(src_output_path, tgt_output_path, src_images, tgt_images):
    common_name_dict = {}
    # 构造一个dict，只为查询方便
    tgt_dict = {}
    for key, value in tgt_images.items():
        tgt_dict[value.name[0:-4]] = get_image_position(value)
    for key, value in src_images.items():
        if value.name[0:-4] in tgt_dict:
            src_xyz = get_image_position(value)
            tgt_xyz = tgt_dict[value.name[0:-4]]
            common_name_dict[value.name[0:-4]] = [src_xyz, tgt_xyz]
    print('common image number = ', len(common_name_dict))
    write_common_name_dict(src_output_path, tgt_output_path, common_name_dict)
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', required=True)
    parser.add_argument('--tgt_path', required=True)
    parser.add_argument('--src_output_path', required=True)
    parser.add_argument('--tgt_output_path', required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print('src_output_path = ', args.src_output_path)
    print('tgt_output_path = ', args.tgt_output_path)
    src = ''
    tgt = ''
    if os.path.isdir(args.src_path):
        print('src is colmap-model')
        src = 'model'
        _, src_images, _ = auto_read_model(args.src_path)
    elif os.path.isfile(args.src_path):
        print('src is colmap-geo-txt')
        src = 'geo'
        src_geo_dict = read_geo(args.src_path)
    else:
        print('src-Error--->', args.src_path)
        exit(0)
    if os.path.isdir(args.tgt_path):
        print('tgt is colmap-model')
        tgt = 'model'
        _, tgt_images, _ = auto_read_model(args.tgt_path)
    elif os.path.isfile(args.tgt_path):
        print('tgt is colmap-geo-txt')
        tgt = 'geo'
        tgt_geo_dict = read_geo(args.tgt_path)
    else:
        print('tgt-Error--->', args.tgt_path)
        exit(0)
    
    if src == 'model' and tgt == 'model':
        images_and_images_to_tum_traj(args.src_output_path, args.tgt_output_path, src_images, tgt_images)
    elif src == 'geo' and tgt == 'geo':
        geo_and_geo_to_tum_traj(args.src_output_path, args.tgt_output_path, src_geo_dict, tgt_geo_dict)
    elif src == 'geo' and tgt == 'model':
        geo_and_images_to_tum_traj(args.src_output_path, args.tgt_output_path, src_geo_dict, tgt_images)
    elif src == 'model' and tgt == 'geo':
        images_and_geo_to_tum_traj(args.src_output_path, args.tgt_output_path, src_images, tgt_geo_dict)

if __name__ == "__main__":
    main()