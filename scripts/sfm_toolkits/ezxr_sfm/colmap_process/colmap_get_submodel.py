# coding: utf-8
import os
import sys
import math
import numpy as np
import shutil
import argparse
sys.path.append('../')
from colmap_process.colmap_model_analyzer import get_folders_in_folder
from colmap_process.colmap_keyframe_selecter import auto_read_model

def run_image_deleter(colmap_exe, input_path, output_path, image_ids_path, image_names_path):
    if not os.path.isfile(image_ids_path) and not os.path.isfile(image_names_path):
        print('Error! image_ids_path or image_names_path at least one should be valid!')
        exit(0)
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    run_str = colmap_exe + ' image_deleter ' + \
        ' --input_path ' + input_path + \
        ' --output_path ' + output_path
    
    if os.path.isfile(image_ids_path):
        run_str = run_str + ' --image_ids_path ' + image_ids_path
    if os.path.isfile(image_names_path):
        run_str = run_str + ' --image_names_path ' + image_names_path
    
    print(run_str)
    os.system(run_str)
    return

def complement_set_of_deleter(input_path, image_ids_path, image_names_path):
    if not os.path.isfile(image_ids_path) and not os.path.isfile(image_names_path):
        print('Error! image_ids_path or image_names_path at least one should be valid!')
        exit(0)
    # 读model，把ids和names都读出来，作为全集
    _, images, _ = auto_read_model(input_path)
    image_ids_full_set = set()
    image_names_full_set = set()
    for key, value in images.items():
        image_ids_full_set.add(key)
        image_names_full_set.add(value.name)
    
    # 读输入的文件，作为子集，再求全集的补集，保存补集文件
    image_ids_output_path = ''
    if os.path.isfile(image_ids_path):
        image_ids_input_set = set()
        fo = open(image_ids_path, "r")
        for line in fo.readlines(): 
            line = line.strip()
            image_ids_input_set.add(line)
        fo.close()
        image_ids_complement_set = image_ids_full_set.difference(image_ids_input_set)
        image_ids_list = list(image_ids_complement_set)
        image_ids_list = sorted(image_ids_list)
        image_ids_output_path = image_ids_path[0:-4] + '_complement_set.txt'
        fo = open(image_ids_output_path, "w")
        for image_id in image_ids_list:
            fo.write(image_id + '\n')
        fo.close()
    
    image_names_output_path = ''
    if os.path.isfile(image_names_path):
        image_names_input_set = set()
        fo = open(image_names_path, "r")
        for line in fo.readlines(): 
            line = line.strip()
            image_names_input_set.add(line)
        fo.close()
        image_names_complement_set = image_names_full_set.difference(image_names_input_set)
        image_names_list = list(image_names_complement_set)
        image_names_list = sorted(image_names_list)
        image_names_output_path = image_names_path[0:-4] + '_complement_set.txt'
        fo = open(image_names_output_path, "w")
        for image_name in image_names_list:
            fo.write(image_name + '\n')
        fo.close()
    return image_ids_output_path, image_names_output_path

def run_image_extractor(colmap_exe, input_path, output_path, image_ids_path, image_names_path):
    image_ids_output_path, image_names_output_path = complement_set_of_deleter(input_path, image_ids_path, image_names_path)
    run_image_deleter(colmap_exe, input_path, output_path, image_ids_output_path, image_names_output_path)
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--colmap_exe', required=True)
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--image_ids_path', type=str, default='')
    parser.add_argument('--image_names_path', type=str, default='')
    parser.add_argument('--use_image_deleter', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.use_image_deleter:
        run_image_deleter(args.colmap_exe, args.input_path, args.output_path, args.image_ids_path, args.image_names_path)
    else:
        run_image_extractor(args.colmap_exe, args.input_path, args.output_path, args.image_ids_path, args.image_names_path)
    return

if __name__ == '__main__':
    main()