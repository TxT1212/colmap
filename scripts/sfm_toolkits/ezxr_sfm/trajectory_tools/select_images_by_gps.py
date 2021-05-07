# coding: utf-8
import os
import sys
import math
import numpy as np
import argparse
import shutil
import utm
import exif
from shapely.geometry import Polygon, Point  # 多边形几何计算
sys.path.append('../')
from colmap_process.create_colmap_custom_match_file_list import searchDirFile
from colmap_process.create_file_list import write_image_list
from trajectory_tools.colmap_to_tum_evo import read_geo
def DDD_to_DMS(ddd):
    '''
    经纬度:“度”转换成“度分秒”
    '''
    dd_str = str(ddd).split('.') # 取“度”的小数部分
    dd = float(dd_str[0]) # 整数部分为“度”
    mm_str = str(float(dd_str[1]) * 60.0).split('.') # 取“分”的小数部分
    mm = float(mm_str[0]) # 整数部分为“分”
    ss = float(mm_str[1]) * 60.0 # 直接是"秒"
    return (dd, mm, ss)

def DMS_to_DDD(dms):
    '''
    经纬度:“度分秒”转换成“度”
    '''
    return dms[0] + dms[1] / 60.0 + dms[2] / 3600.0

def write_geo_file(geo_path, geo_dict):
    fid = open(geo_path, "w")
    for key, value in geo_dict.items():
        line = key + ' ' + str(value[0]) + ' ' + str(value[1]) + ' ' + str(value[2]) + '\n'
        fid.write(line)
    fid.close()
    return

def read_image_gps_as_wgs84(image_path):
    with open(image_path, 'rb') as image_file: # 一定要用binary的方式打开图像
        my_image = exif.Image(image_file)
        if not my_image.has_exif:
            print('Warnning, ', image_path, ' has no exif!')
            return None, None, None
        lat_deg = DMS_to_DDD(my_image.gps_latitude)
        lon_deg = DMS_to_DDD(my_image.gps_longitude)
        alt_m = my_image.gps_altitude
    return (lat_deg, lon_deg, alt_m)

def read_images_in_folder(image_folder):
    if image_folder[-1] != '/':
        image_folder = image_folder + '/'
    exts = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG']
    results = searchDirFile(image_folder, exts)
    images_list = []
    for res in results:
        name = res.replace(image_folder, '')
        images_list.append(name)
    return images_list

def read_boundary_as_polygon(boundary_file_path):
    '''
    说明: 多边形必须在utm坐标系下，因为经纬度是球形坐标，非欧式空间
    '''
    infile = open(boundary_file_path, 'r')
    first_line = infile.readline().strip()
    first_line_strs = first_line.split(':')
    assert len(first_line_strs) == 2, 'Error, the first line should be: # type:WGS84 or # type:UTM'
    boundary_bb = np.loadtxt(boundary_file_path, delimiter=',')
    if first_line_strs[1] == 'WGS84' or first_line_strs[1] == 'wgs84':
        for idx in range(boundary_bb.shape[0]):
            utm_info = utm.from_latlon(boundary_bb[idx][0], boundary_bb[idx][1])
            boundary_bb[idx][0] = utm_info[0]
            boundary_bb[idx][1] = utm_info[1]
    elif first_line_strs[1] == 'UTM' or first_line_strs[1] == 'utm':
        pass
    else:
        assert False, 'Error, the first line should be: # type:WGS84 or # type:UTM'
    assert(boundary_bb.shape[0] > 2) # 至少是三角形吧
    pts = []
    for idx in range(boundary_bb.shape[0]):
        pts.append(tuple(boundary_bb[idx,0:2])) # 只取xy
    poly = Polygon(pts)
    return poly

def select_images_by_boundary(boundary_file_path, offset_path, image_folder, images_list_outpath, geo_outpath):
    poly = read_boundary_as_polygon(boundary_file_path)
    offset_xy = [0.0, 0.0]
    if offset_path != '':
        offset_xy = np.loadtxt(offset_path)

    images_list = read_images_in_folder(image_folder)

    images_dict_in_boundary = {}
    images_dict_out_boundary = {}
    images_list_in_boundary = []
    images_list_out_boundary = []
    idx = 0
    delta = int(len(images_list) * 0.02) * 10
    for img_rel_path in images_list:
        if idx % delta == 0:
            print('processed/total: ', idx, '/', len(images_list))
        idx += 1
        img_gps = read_image_gps_as_wgs84(os.path.join(image_folder, img_rel_path))
        if img_gps[0] is None:
            continue
        utm_info = utm.from_latlon(img_gps[0], img_gps[1])
        # 减去offset
        east_m = utm_info[0] - offset_xy[0]
        north_m = utm_info[1] - offset_xy[1]
        pt_tocheck = Point(east_m, north_m)
        if poly.contains(pt_tocheck):
            images_dict_in_boundary[img_rel_path] = [east_m, north_m, img_gps[2]]
            images_list_in_boundary.append(img_rel_path)
        else:
            images_dict_out_boundary[img_rel_path] = [east_m, north_m, img_gps[2]]
            images_list_out_boundary.append(img_rel_path)
    # 都排个序
    images_dict_in_boundary = dict(sorted(images_dict_in_boundary.items(), key=lambda item: item[0]))
    images_list_in_boundary = sorted(images_list_in_boundary)
    images_dict_out_boundary = dict(sorted(images_dict_out_boundary.items(), key=lambda item: item[0]))
    images_list_out_boundary = sorted(images_list_out_boundary)
    # 生成geos.txt用于后续evo检查
    print('save geo_file to ', geo_outpath)
    out_geo_outpath = geo_outpath[0:-4] + '_out.txt'
    write_geo_file(geo_outpath, images_dict_in_boundary)
    write_geo_file(out_geo_outpath, images_dict_out_boundary)
    # 生成image_list.txt
    print('save image_list_file to ', images_list_outpath)
    out_images_list_outpath = images_list_outpath[0:-4] + '_out.txt'
    write_image_list(images_list_outpath, images_list_in_boundary)
    write_image_list(out_images_list_outpath, images_list_out_boundary)
    print('numbers in/all = ', len(images_dict_in_boundary), '/', len(images_list))
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--boundary_file_path', required=True)
    parser.add_argument('--offset_path', default='', type=str)
    parser.add_argument('--image_folder', required=True)
    parser.add_argument('--images_list_outpath', required=True)
    parser.add_argument('--geo_outpath', required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    select_images_by_boundary(args.boundary_file_path, args.offset_path, args.image_folder, args.images_list_outpath, args.geo_outpath)
    
if __name__ == "__main__":
    main()