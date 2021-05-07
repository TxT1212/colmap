# coding: utf-8
import cv2
import os
import argparse
import time
import sys

sys.path.append('../')
from colmap_process.create_file_list import write_image_list
from colmap_process.colmap_seq_sfm import run_seq_sfm
from colmap_process.colmap_model_analyzer import get_folders_in_folder
def run_colmap_seq_sfms(colmap_exe, image_folder, database_folder, model_folder, seq_list, seq_run_mapper_flags=None, suffix_str='',
                seq_match_overlap=20, mapper_min_num_matches=45, 
                mapper_init_min_num_inliers=200, mapper_abs_pose_min_num_inliers=60,
                mask_path='',
                optionalMapper=False):
    print('colmap_exe = ', colmap_exe)
    print('image_folder = ', image_folder)
    print('database_folder = ', database_folder)
    print('model_folder = ', model_folder)

    # 图像文件夹不存在，直接返回错误
    if not os.path.isdir(image_folder):
        print('Error image_folder:', image_folder)
        exit(0)
    
    # 文件路径字符统一
    if image_folder[-1] != '/':
        image_folder = image_folder + '/'
    
    if database_folder[-1] != '/':
        database_folder = database_folder + '/'

    if model_folder[-1] != '/':
        model_folder = model_folder + '/'
    
    # 文件是否存在, 不存在创建
    if not os.path.isdir(database_folder):
        os.mkdir(database_folder)
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)
    
    # -------------------------------------------------
    # 现在支持外部传进来的list变量
    # -------------------------------------------------
    img_folder_name_list = None
    if seq_list is not None:
        # 直接把它赋给循环变量
        img_folder_name_list = seq_list
    else:
        # 循环images下的所有文件夹
        img_folder_name_list = get_folders_in_folder(image_folder)
    # 主循环
    for folder in img_folder_name_list:
        database_path = database_folder + folder + suffix_str + '.db'
        image_list_path = image_folder + folder + suffix_str + '.txt'
        output_path = model_folder + folder + suffix_str
        # 如果image list不存在就不跑它
        if not os.path.isfile(image_list_path):
            print('Warning! Skip! no file named: ', image_list_path)
            continue
        # 具体的model文件夹如果不存在就创建
        if not os.path.isdir(output_path):
            os.mkdir(output_path)

        if not seq_run_mapper_flags == None:
            optionalMapperCurSeq = (optionalMapper and (not seq_run_mapper_flags[folder]))
        else:
            optionalMapperCurSeq = False

        run_seq_sfm(colmap_exe, database_path, image_folder, image_list_path, output_path, \
                seq_match_overlap, mapper_min_num_matches, mapper_init_min_num_inliers, mapper_abs_pose_min_num_inliers,
                mask_path=mask_path,
                optionalMapper=optionalMapperCurSeq)
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--colmap_exe', required=True)
    parser.add_argument('--image_folder', required=True)
    parser.add_argument('--database_folder', required=True)
    parser.add_argument('--model_folder', required=True)
    parser.add_argument('--is_kf', default=False)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    run_colmap_seq_sfms(args.colmap_exe, args.image_folder, args.database_folder, args.model_folder, None, args.is_kf)
    
if __name__ == '__main__':
    main()