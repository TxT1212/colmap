# coding: utf-8
import os
import sys
import math
import numpy as np
import argparse
import shutil

sys.path.append('../')
from colmap_process.colmap_db_parser import *
from colmap_process.colmap_read_write_model import *

from colmap_process.colmap_keyframe_selecter import auto_read_model

def get_folders_in_folder(model_path):
    if model_path[-1] != '/':
        model_path = model_path + '/'
    model_folders = os.listdir(model_path) # model文件夹下, 默认输出的应该是文件夹0, 1, 2...,代表多个model
    folders = []
    for model_folder in model_folders:
        if os.path.isdir(model_path + model_folder):
            folders.append(model_folder)
    return folders

def image_folder_histogram_in_databse(imgs):
    img_folder_hist = {}
    for img in imgs:
        strs = img[1].split('/')
        if img_folder_hist.get(strs[0]) is not None:
            img_folder_hist[strs[0]] += 1
        else:
            img_folder_hist[strs[0]] = 0
    return img_folder_hist

def image_folder_histogram_in_model(imgs):
    img_folder_hist = {}
    for key, img in imgs.items():
        strs = img.name.split('/')
        if img_folder_hist.get(strs[0]) is not None:
            img_folder_hist[strs[0]] += 1
        else:
            img_folder_hist[strs[0]] = 0
    return img_folder_hist

def compute_individual_img_ratio(img_hist_db, img_hist_model):
    img_hist_ratio = {}
    for key, value in img_hist_db.items():
        if img_hist_model.get(key) is not None:
            if value > 0.0:
                img_hist_ratio[key] = float(img_hist_model[key]) / value
            else:
                img_hist_ratio[key] = 0
        else:
            img_hist_ratio[key] = 0.0
    return img_hist_ratio

def get_database_info(database_path):
    db = COLMAPDatabase.connect(database_path)
    cursor = db.cursor()
    cursor.execute("select * from cameras")
    results = cursor.fetchall()
    print('cameras number:', len(results))
    camera_num = len(results)

    cursor = db.cursor()
    cursor.execute("select * from images")
    results = cursor.fetchall()
    print('images number:', len(results))
    image_num = len(results)
    img_hist_db = image_folder_histogram_in_databse(results)

    cursor = db.cursor()
    cursor.execute("select * from keypoints")
    results = cursor.fetchall()
    print('keypoints number:', len(results))
    keypoints_num = len(results)
    db.close()

    return camera_num, image_num, keypoints_num, img_hist_db

def compute_num_observations(images):
    '''
    计算每张图像观测到的(有效)3d点的累加和
    注意是,图像,观测,(有效)3d点
    '''
    valid_pt3d_num = 0
    for _, image in images.items():
        # 没有观测到3d点的2d点, 值是-1
        valid_pt3d_num = valid_pt3d_num + len([i for i in image.point3D_ids if i > 0])
    return valid_pt3d_num

def compute_mean_observations_per_image(valid_pt3d_num, images):
    return valid_pt3d_num * 1.0 / len(images)

def compute_mean_track_length(valid_pt3d_num, points3d):
    if len(points3d) == 0:
        return -1
    return valid_pt3d_num * 1.0 / len(points3d)

def compute_valid_img_ratio(image_num_in_db, images):
    return len(images) * 1.0 / image_num_in_db

def compute_mean_reprojection_error(points3d):
    if (len(points3d) == 0):
        return -1
    sum_error = 0
    sum_image = 0
    # 每个3d点的error存储的是它观测到的所有反投影误差的平均值
    for _, pt3d in points3d.items():
        sum_error += pt3d.error * len(pt3d.image_ids)
        sum_image += len(pt3d.image_ids)
    return sum_error * 1.0 / sum_image

def run_model_analyzer(database_path, model_path, 
                    mean_obs_per_img_threshold, mean_track_length_threshold, 
                    valid_img_ratio_threshold, mean_rep_error_threshold):
    statistic_info = np.zeros(4)
    if model_path != '/':
        model_path = model_path + '/'
    report_path = model_path + 'model_report.txt'
    # 如果该model依赖的子节点已经失败了，外部不会创建该model文件夹
    # 这里需要建一个新文件夹，用来放report
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    # 如果连database都没有，说明跳过了这个节点，直接返回一个失败的report文件
    if not os.path.isfile(database_path):
        print("warnning! no-database in --->", database_path)
        status_str = 'no-databse'
        write_model_analyze_report(report_path, status_str, statistic_info, None)
        return statistic_info, None, status_str
    model_folders = get_folders_in_folder(model_path)
    status_str = 'success'
    # 读database
    camera_num, image_num, keypoints_num, img_hist_db = get_database_info(database_path)
    # 异常判断
    # 如果文件夹下有多个models, 循环所有models, 如果有一个model的有效图像数 > valid_img_ratio_threshold, 说明该model有效; 否则失败
    if len(model_folders) > 1: 
        print("warnning! multi-models in --->", model_path)
        # 遍历所有models，计算他们的valid_img_ratio，保留最大的那个
        valid_model_folder = None
        valid_img_ratio = 0.0
        for model_folder in model_folders:
            cameras, images, points3d = auto_read_model(model_path + model_folder)
            valid_img_ratio_test = compute_valid_img_ratio(image_num, images)
            if valid_img_ratio < valid_img_ratio_test:
                valid_img_ratio = valid_img_ratio_test
                valid_model_folder = model_folder
        # 如果model有效图像数百分比过小， 就返回失败
        if valid_img_ratio < valid_img_ratio_threshold:
            status_str = 'multi-models'
            statistic_info[2] = valid_img_ratio
            write_model_analyze_report(report_path, status_str, statistic_info, None)
            return statistic_info, None, status_str
        # 如果model的有效图像数百分比还行， 进行文件操作， 把无效model的文件夹删除， 把有效model的文件夹改名成0
        print("Good news! valid-model in multi-models --->", model_path)
        for model_folder in model_folders:
            if model_folder != valid_model_folder:
                shutil.rmtree(model_path + model_folder)
        # 如果有效的model文件夹不是0，把有效的model文件夹名字改成0
        if valid_model_folder != '0':
            src_path = model_path + valid_model_folder
            dst_path = model_path + '0'
            os.rename(src_path, dst_path)
    # 没有任何model生成，直接返回
    if len(model_folders) == 0:
        print("warnning! no-model in --->", model_path)
        status_str = 'no-model'
        write_model_analyze_report(report_path, status_str, statistic_info, None)
        return statistic_info, None, status_str
    # -------------------------------------------------------------------------------
    # 正常处理
    # 读model
    cameras, images, points3d = auto_read_model(model_path + '0')
    img_hist_model = image_folder_histogram_in_model(images)
    # 统计
    valid_pt3d_num = compute_num_observations(images)
    mean_obs_per_img = compute_mean_observations_per_image(valid_pt3d_num, images)
    mean_track_length = compute_mean_track_length(valid_pt3d_num, points3d)
    valid_img_ratio = compute_valid_img_ratio(image_num, images)
    mean_rep_error = compute_mean_reprojection_error(points3d)
    # 对每个子节点单独统计
    img_hist_ratio = compute_individual_img_ratio(img_hist_db, img_hist_model)
    # 汇总统计
    statistic_info[0] = mean_obs_per_img
    statistic_info[1] = mean_track_length
    statistic_info[2] = valid_img_ratio
    statistic_info[3] = mean_rep_error
    # 汇总统计有效性判断
    if mean_obs_per_img < mean_obs_per_img_threshold or \
        mean_track_length < mean_track_length_threshold or \
        valid_img_ratio < valid_img_ratio_threshold or \
        mean_rep_error > mean_rep_error_threshold:
        status_str = 'failed'
    # 对每个子节点单独有效性判断
    # total_hist_ratio = 0.0
    # sum_hist_ratio = 0.0
    # for _, registed_ratio in img_hist_ratio.items():
    #     total_hist_ratio += 1.0
    #     # 统计子节点的有效百分比，如果小于阈值，就是0，大于等于就是1
    #     if registed_ratio >= valid_img_ratio_threshold:
    #         sum_hist_ratio += 1.0
    # img_hist_ratio_percent = sum_hist_ratio / total_hist_ratio
    # if img_hist_ratio_percent < valid_img_ratio_threshold:
    #     status_str = 'seq-invalid'
    write_model_analyze_report(report_path, status_str, statistic_info, img_hist_ratio)
    return statistic_info, img_hist_ratio, status_str

def write_model_analyze_report(report_path, status_str, statistic_info, img_hist_ratio):
    print('write to --->', report_path)
    geo_file = open(report_path,'w')
    geo_line = '# status:' + status_str + '\n'
    print(geo_line)
    geo_file.write(geo_line)
    geo_line = '# mean_obs_per_img:' + str(format(statistic_info[0], '.2f')) + '\n'
    print(geo_line)
    geo_file.write(geo_line)
    geo_line = '# mean_track_length:' + str(format(statistic_info[1], '.2f')) + '\n'
    print(geo_line)
    geo_file.write(geo_line)
    geo_line = '# valid_img_ratio:' + str(format(statistic_info[2], '.2f')) + '\n'
    print(geo_line)
    geo_file.write(geo_line)
    geo_line = '# mean_rep_error:' + str(format(statistic_info[3], '.2f')) + '\n'
    print(geo_line)
    geo_file.write(geo_line)
    geo_line = '# img_hist_ratio:\n'
    if img_hist_ratio is not None:
        for key, value in img_hist_ratio.items():
            geo_line = geo_line + '# ' + key + ' = ' + str(format(value, '.2f')) + '\n'
    geo_file.write(geo_line)
    geo_file.close()
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_path', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--mean_obs_per_img_threshold', type=float, default=150.0)
    parser.add_argument('--mean_track_length_threshold', type=float, default=3.0)
    parser.add_argument('--valid_img_ratio_threshold', type=float, default=0.90)
    parser.add_argument('--mean_rep_error_threshold', type=float, default=4.0)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    run_model_analyzer(args.database_path, args.model_path, 
                args.mean_obs_per_img_threshold, args.mean_track_length_threshold, 
                args.valid_img_ratio_threshold, args.mean_rep_error_threshold)
    return

if __name__ == '__main__':
    main()