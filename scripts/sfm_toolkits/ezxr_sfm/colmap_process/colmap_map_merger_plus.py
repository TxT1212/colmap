# coding: utf-8
import os
import sys
import math
import numpy as np
import argparse

sys.path.append('../')
from colmap_process.colmap_seq_sfm import run_custom_matcher, run_mapper, run_image_registrator, run_model_aligner, run_model_merger
from colmap_process.colmap_model_modify import update_loc_model_id_refer_to_locmap_database, update_loc_model_id_refer_to_map_model_id
# from colmap_seq_sfm_ransac_sim3 import seq_sfm_ransac_sim3
from colmap_process.colmap_keyframe_selecter import auto_read_model
from colmap_process.colmap_get_submodel import run_image_extractor

def run_database_merger(colmap_exe, loc_db, map_db, locmap_db):
    '''
    database合并, 调用colmap的命令行
    '''
    run_str = colmap_exe + ' database_merger --database_path1 ' + map_db + \
        ' --database_path2 ' + loc_db + \
        ' --merged_database_path ' + locmap_db
    print(run_str)
    os.system(run_str)
    return

def common_images_of_two_models(loc_model_folder, map_model_folder, common_images_output_path):
    '''
    寻找两个model的公共图像，并输出到image_list.txt
    '''
    _, loc_images, _ = auto_read_model(loc_model_folder)
    _, map_images, _ = auto_read_model(map_model_folder)
    loc_set = set()
    map_set = set()
    for _, value in loc_images.items():
        loc_set.add(value.name)
    for _, value in map_images.items():
        map_set.add(value.name)
    common_images_set = map_set.intersection(loc_set)
    common_images_list = list(common_images_set)
    common_images_list = sorted(common_images_list)
    fo = open(common_images_output_path, "w")
    for image_name in common_images_list:
        fo.write(image_name + '\n')
    fo.close()
    return

def run_submodel_merger(colmap_exe, loc_model_folder, map_model_folder, output_model_folder, sim3_max_reproj_error):
    '''
    寻找两个model的公共图像，并输出到两个submodel
    修改loc_submodel的id，使它跟map_submodel一致
    调用model_merger，计算sim3.txt
    '''
    # map_model_folder和loc_model_folder都是在0那一级的目录
    if not os.path.exists(output_model_folder):
        os.mkdir(output_model_folder)
    print('output_model_folder: ', output_model_folder)
    common_images_output_path = os.path.join(output_model_folder, 'common_images_list.txt')
    map_submodel_folder = os.path.join(output_model_folder, 'map_submodel')
    loc_submodel_folder = os.path.join(output_model_folder, 'loc_submodel')
    loc_id_modified_submodel_folder = os.path.join(output_model_folder, 'id_modified_submodel')
    mergered_submodel_folder = os.path.join(output_model_folder, 'mergered_submodel')
    if not os.path.exists(map_submodel_folder):
        os.mkdir(map_submodel_folder)
    if not os.path.exists(loc_submodel_folder):
        os.mkdir(loc_submodel_folder)
    if not os.path.exists(loc_id_modified_submodel_folder):
        os.mkdir(loc_id_modified_submodel_folder)
    if not os.path.exists(mergered_submodel_folder):
        os.mkdir(mergered_submodel_folder)
    # 寻找两个model的公共图像
    print('common_images_of_two_models...')
    common_images_of_two_models(loc_model_folder, map_model_folder, common_images_output_path)
    print('extract common submodel from loc and map model...')
    run_image_extractor(colmap_exe, map_model_folder, map_submodel_folder, '', common_images_output_path)
    run_image_extractor(colmap_exe, loc_model_folder, loc_submodel_folder, '', common_images_output_path)
    print('update_loc_model_id_refer_to_map_model_id...')
    update_loc_model_id_refer_to_map_model_id(loc_submodel_folder, map_submodel_folder, loc_id_modified_submodel_folder)
    print('run_model_merger...')
    run_model_merger(colmap_exe, loc_id_modified_submodel_folder, map_submodel_folder, mergered_submodel_folder, max_reproj_error=sim3_max_reproj_error)
    return

def run_map_merger_pipeline(colmap_exe, image_path, image_list_path, match_list_path, 
                        database_path_loc, database_path_map, database_path_locmap, 
                        model_folder_loc, model_folder_new_loc, model_folder_map, model_folder_registrator, model_folder_merge, 
                        mapper_abs_pose_min_num_inliers=60, sim3_max_reproj_error=16, use_mapper=False):
    # database merge
    print('run_database_merger...')
    print('database_path_map = ', database_path_map)
    run_database_merger(colmap_exe, database_path_loc, database_path_map, database_path_locmap)
    # custom match
    print('run_custom_matcher...')
    run_custom_matcher(colmap_exe, database_path_locmap, match_list_path, min_num_inliers=15)
    # image registrator
    print('model_folder_registrator = ', model_folder_registrator)
    if use_mapper:
        # 用mapper的方式实现更高精度的registrator
        print('run_image_registrator(mapper)...')
        run_mapper(colmap_exe, database_path_locmap, image_path, image_list_path, model_folder_registrator, \
            min_num_matches = 15, init_min_num_inliers = 100, abs_pose_min_num_inliers=mapper_abs_pose_min_num_inliers, 
            mapper_input_model_path=model_folder_map,
            mapper_ba_global_images_ratio=3.0, mapper_ba_global_points_ratio=3.0, mapper_fix_existing_images=1)
        # mapper_fix_existing_images不会fix尺度，所以还要把尺度还原
        run_model_aligner(colmap_exe, model_folder_registrator+'/0', model_folder_map + '/geos.txt', model_folder_registrator+'/0', max_error=0.05)
    else:
        print('run_image_registrator...')
        run_image_registrator(colmap_exe, database_path_locmap, model_folder_map, model_folder_registrator, 
                    min_num_matches=15, init_min_num_inliers=100, abs_pose_min_num_inliers=mapper_abs_pose_min_num_inliers)
    # update
    print('update_loc_model_id_refer_to_locmap_database...')
    update_loc_model_id_refer_to_locmap_database(model_folder_loc, database_path_locmap, model_folder_new_loc)
    # model merge
    print('run_model_merger...')
    run_model_merger(colmap_exe, model_folder_new_loc, model_folder_registrator+'/0', model_folder_merge, max_reproj_error=sim3_max_reproj_error)
    return

def parse_args():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--colmap_exe', required=True)
    # parser.add_argument('--match_list_path', required=True)
    # parser.add_argument('--image_path', required=True)

    # parser.add_argument('--database_path_loc', required=True)
    # parser.add_argument('--database_path_map', required=True)
    # parser.add_argument('--database_path_locmap', required=True)

    # parser.add_argument('--model_folder_loc', required=True)
    # parser.add_argument('--model_folder_new_loc', required=True)
    # parser.add_argument('--model_folder_map', required=True)
    # parser.add_argument('--model_folder_registrator', required=True)
    # parser.add_argument('--model_folder_merge', required=True)

    # parser.add_argument('--mapper_abs_pose_min_num_inliers', type=int, default=60)
    # parser.add_argument('--sim3_max_reproj_error', type=int, default=16)
    # parser.add_argument('--use_mapper', action='store_true')
    # args = parser.parse_args()

    parser = argparse.ArgumentParser()
    parser.add_argument('--colmap_exe', required=True)

    parser.add_argument('--model_folder_loc', required=True)
    parser.add_argument('--model_folder_map', required=True)
    parser.add_argument('--model_folder_output', required=True)

    parser.add_argument('--sim3_max_reproj_error', type=int, default=16)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # run_map_merger_pipeline(args.colmap_exe, args.image_path, '', args.match_list_path, 
    #                     args.database_path_loc, args.database_path_map, args.database_path_locmap, 
    #                     args.model_folder_loc, args.model_folder_new_loc, args.model_folder_map, args.model_folder_registrator, args.model_folder_merge, 
    #                     args.mapper_abs_pose_min_num_inliers, args.sim3_max_reproj_error, args.use_mapper)
    run_submodel_merger(args.colmap_exe, args.model_folder_loc, args.model_folder_map, args.model_folder_output, args.sim3_max_reproj_error)
    return

if __name__ == '__main__':
    main()