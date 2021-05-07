# coding: utf-8
import os
import sys
import math
import numpy as np
import argparse
import shutil
import utm
sys.path.append('../')
from colmap_process.colmap_db_parser import *

def write_geo_format(geo_path, geo_dict):
    first_key = ''
    for key, value in geo_dict.items():
        first_key = key
        break
    offset_x = geo_dict[first_key][0]
    offset_y = geo_dict[first_key][1]
    offset_geo_path = geo_path[0:-4] + '_offset.txt'
    offset_file = open(offset_geo_path,'w')
    line = str(offset_x) + ' ' + str(offset_y)
    offset_file.write(line)
    offset_file.close()
    fid = open(geo_path, "w")
    for key, value in geo_dict.items():
        line = key + ' ' + str(value[0] - offset_x) + ' ' + str(value[1] - offset_y) + ' ' + str(value[2]) + '\n'
        fid.write(line)
    fid.close()
    return

def read_db_as_geo_format(database_path):
    '''
    0 image_id 
    1 name 
    2 camera_id
    3 prior_qw
    4 prior_qx
    5 prior_qy
    6 prior_qz
    7 prior_tx
    8 prior_ty
    9 prior_tz
    '''
    db = COLMAPDatabase.connect(database_path)
    cursor = db.cursor()
    cursor.execute("select * from images")
    imgs = cursor.fetchall()
    print('images number:', len(imgs))

    geo_dict = {}
    for img in imgs:
        lat = float(img[7])
        lon = float(img[8])
        alt = float(img[9])
        utm_info = utm.from_latlon(lat, lon)
        easting = utm_info[0]
        northing = utm_info[1]
        geo_dict[img[1]] = [easting, northing, alt]

    db.close()
    return geo_dict

def db_to_geo(database_path, geo_path):
    geo_dict = read_db_as_geo_format(database_path)
    write_geo_format(geo_path, geo_dict)
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_path', required=True)
    parser.add_argument('--geo_path', required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    db_to_geo(args.database_path, args.geo_path)
    return

if __name__ == '__main__':
    main()