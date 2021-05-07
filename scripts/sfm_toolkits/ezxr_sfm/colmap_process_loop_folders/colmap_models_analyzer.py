# coding: utf-8
import cv2
import os
import argparse
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
# from textwrap import wrap
sys.path.append('../')
from colmap_process.colmap_model_analyzer import run_model_analyzer, get_folders_in_folder

def visualize_horizontal_bar(model_folder, sub_model_folders, statistic_list, title_str, max_limit, suffix_str='', suffix_str2=''):
    assert(len(sub_model_folders) == statistic_list.shape[0])
    fig, ax = plt.subplots(figsize=(10, 20))
    plt.grid(True)
    fig.tight_layout()
    # labels = [ '\n'.join(wrap(l, 40)) for l in sub_model_folders ]
    y_pos = np.arange(len(sub_model_folders))
    ax.barh(y_pos, statistic_list)
    plt.yticks(y_pos, sub_model_folders)
    ax.set_title(title_str)
    ax.set_xlim([0, max_limit])
    ax.invert_yaxis()  # labels read top-to-bottom
    save_str = model_folder + title_str + suffix_str + suffix_str2 + '.png'
    plt.savefig(save_str, dpi=500, bbox_inches='tight')
    plt.close()
    return

def visualize_model_statistics(model_folder, sub_model_folders, statistic_info_list, suffix_str='', suffix_str2=''):
    '''
    statistic_info[0] = mean_obs_per_img
    statistic_info[1] = mean_track_length
    statistic_info[2] = valid_img_ratio
    statistic_info[3] = mean_rep_error
    '''
    mean_obs_per_img_list = statistic_info_list[:, 0]
    mean_track_length_list = statistic_info_list[:, 1]
    valid_img_ratio_list = statistic_info_list[:, 2]
    mean_rep_error_list = statistic_info_list[:, 3]
    print('draw statistics bar images ... ')
    visualize_horizontal_bar(model_folder, sub_model_folders, mean_obs_per_img_list, 'mean_obs_per_img', 1000, suffix_str, suffix_str2)
    visualize_horizontal_bar(model_folder, sub_model_folders, mean_track_length_list, 'mean_track_length', 20, suffix_str, suffix_str2)
    visualize_horizontal_bar(model_folder, sub_model_folders, valid_img_ratio_list, 'valid_img_ratio', 1.0, suffix_str, suffix_str2)
    visualize_horizontal_bar(model_folder, sub_model_folders, mean_rep_error_list, 'mean_rep_error', 4.0, suffix_str, suffix_str2)
    return

def run_colmap_models_analyzer(database_folder, model_folder, todo_list=None, suffix_str='', suffix_str2='', 
                mean_obs_per_img_threshold=150.0, mean_track_length_threshold=3.0, 
                valid_img_ratio_threshold=0.9, mean_rep_error_threshold=2.0):
    print('database_folder = ', database_folder)
    print('model_folder = ', model_folder)
    print('mean_obs_per_img_threshold = ', mean_obs_per_img_threshold)
    print('mean_track_length_threshold = ', mean_track_length_threshold)
    print('valid_img_ratio_threshold = ', valid_img_ratio_threshold)
    print('mean_rep_error_threshold = ', mean_rep_error_threshold)
    # 文件路径字符统一
    if database_folder[-1] != '/':
        database_folder = database_folder + '/'

    if model_folder[-1] != '/':
        model_folder = model_folder + '/'
    
    # ------------------------------------------
    # ----------------参数适配-------------------
    sub_model_folders = []
    if todo_list is None: # 如果这个参数是None，那就遍历model的文件夹
        sub_model_folders = get_folders_in_folder(model_folder)
    elif type(todo_list) is list and len(todo_list) > 0: # 如果参数是有效list
        if type(todo_list[0]) is str: # 元素是str，说明是seq或者route
            sub_model_folders = todo_list
        elif type(todo_list[0]) is list and len(todo_list[0]) == 2:# 元素是list，说明是routepair
            for routepair_name in todo_list:
                aa = routepair_name[0]
                bb = routepair_name[1]
                aa_bb = aa + '_' + bb
                sub_model_folders.append(aa_bb)
        else:
            raise Exception('Error todo_list in run_colmap_models_analyzer')
    else:
        raise Exception('Error todo_list in run_colmap_models_analyzer')
    
    statistic_info_list = []
    valid_sub_model_folders = []
    sub_model_status_dict = {}
    for sub_model_folder in sub_model_folders:
        full_database_path = database_folder + sub_model_folder + suffix_str + '.db'
        if not os.path.isfile(full_database_path):
            print('Warning! No database named: ', full_database_path)
            # continue
        full_model_path = model_folder + sub_model_folder + suffix_str
        if not os.path.isdir(full_model_path):
            print('Warning! No model folder named: ', full_model_path)
            # continue
        # print(mean_obs_per_img_threshold, mean_track_length_threshold, valid_img_ratio_threshold, mean_rep_error_threshold)
        statistic_info, _, status_str = run_model_analyzer(full_database_path, full_model_path, \
                mean_obs_per_img_threshold, mean_track_length_threshold, \
                valid_img_ratio_threshold, mean_rep_error_threshold)
        statistic_info_list.append(statistic_info)
        valid_sub_model_folders.append(sub_model_folder)
        sub_model_status_dict[sub_model_folder] = status_str
    # 如果没有一条有效的数据，直接返回
    if len(statistic_info_list) == 0:
        return sub_model_status_dict
    statistic_info_list = np.array(statistic_info_list)
    visualize_model_statistics(model_folder, valid_sub_model_folders, statistic_info_list, suffix_str, suffix_str2)
    return sub_model_status_dict

def read_model_report_status(report_path):
    with open(report_path) as f:
        content = f.readlines()
        if len(content) == 0:
            return None
        first_line = content[0].strip()
        first_line_strs = first_line.split(':')
        return first_line_strs[1]

def read_model_reports(model_folder, route_list=None, routepair_list=None, suffix_str=''):
    if model_folder[-1] != '/':
        model_folder = model_folder + '/'
    sub_model_folders = []
    sub_model_status_dict = {}

    # 如果没有传入额外信息，就遍历文件夹
    if route_list is None or routepair_list is None:
        sub_model_folders = get_folders_in_folder(model_folder)
    else: # 否则就按照给定的route，routepair的文件夹，进行遍历
        for route in route_list:
            route_str = route + suffix_str # 适配是否为_kf
            sub_model_folders.append(route_str)
        for routepair in routepair_list:
            routepair_str = routepair[0] + '_' + routepair[1] + suffix_str # 适配是否为_kf
            sub_model_folders.append(routepair_str)
    
    for sub_model_folder in sub_model_folders:
        full_model_path = model_folder + sub_model_folder
        if not os.path.isdir(full_model_path):
            print('No model folder named: ', full_model_path)
            # 就算没有也要添加key，保证该变量的key是全图
            sub_model_status_dict[sub_model_folder] = None
            continue
        report_path = full_model_path + '/model_report.txt'
        sub_model_status_dict[sub_model_folder] = read_model_report_status(report_path)
    return sub_model_status_dict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_folder', required=True)
    parser.add_argument('--model_folder', required=True)
    parser.add_argument('--mean_obs_per_img_threshold', type=float, default=150.0)
    parser.add_argument('--mean_track_length_threshold', type=float, default=3.0)
    parser.add_argument('--valid_img_ratio_threshold', type=float, default=0.90)
    parser.add_argument('--mean_rep_error_threshold', type=float, default=2.0)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    run_colmap_models_analyzer(args.database_folder, args.model_folder, None, '', '', 
                args.mean_obs_per_img_threshold, args.mean_track_length_threshold, 
                args.valid_img_ratio_threshold, args.mean_rep_error_threshold)

if __name__ == '__main__':
    main()