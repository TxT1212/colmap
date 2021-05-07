# coding: utf-8
import os
import sys
import math
import numpy as np
import shutil
import argparse
import json
sys.path.append('../')
from colmap_process.colmap_model_analyzer import get_folders_in_folder
from colmap_process.colmap_keyframe_selecter import auto_read_model
from colmap_process.colmap_db_parser import *
from colmap_process.colmap_export_geo import colmap_export_geo
from colmap_process.colmap_get_submodel import run_image_extractor
from colmap_process_loop_folders.basic_colmap_operation import readImageList, writeStrList, checkAndDeleteFiles
def run_database_creator(colmap_exe, database_path):
    if os.path.exists(database_path):
        os.remove(database_path)

    run_str = colmap_exe + ' database_creator ' + \
        ' --database_path ' + database_path
    print(run_str)
    os.system(run_str)
    return

def readCameraConfig(jsonFile):
    cameraModel = 'SIMPLE_RADIAL'
    focalLenFactor = 1.2

    if os.path.isfile(jsonFile):
        with open(jsonFile, 'r', encoding='UTF-8') as fp:
            jsonValue = json.load(fp)
        
        cameraModel = jsonValue['cameraModel']
        focalLenFactor = jsonValue['focalLenFactor']

    return cameraModel, focalLenFactor

def classifyImagesBySubFolder(imageListPath, imagePath, cameraConfigPath):
    subImageListFiles = []
    cameraModles = []
    focalLenFactors = []

    imageNames = readImageList(imageListPath)
    subFolderImageNames = {}

    for imageName in imageNames:
        folderName, _ = os.path.split(imageName) 
        folderName = folderName.strip('/').strip('\\')

        if not (folderName in subFolderImageNames.keys()):
            subFolderImageNames[folderName] = []
        
        subFolderImageNames[folderName].append(imageName)
    
    for key in subFolderImageNames.keys():
        subImageListFile = os.path.join(imagePath, key + '_classified_tmp.txt')
        writeStrList(subFolderImageNames[key], subImageListFile)

        cameraModel, focalLenFactor = readCameraConfig(os.path.join(cameraConfigPath, key+'.json'))
        subImageListFiles.append(subImageListFile)
        cameraModles.append(cameraModel)
        focalLenFactors.append(focalLenFactor)

    return subImageListFiles, cameraModles, focalLenFactors

def run_feature_extractor(colmap_exe, database_path, image_path, image_list_path, mask_path=''):
    subImageListFiles, cameraModles, focalLenFactors = \
    classifyImagesBySubFolder(image_list_path, image_path, os.path.join(image_path, 'cameraConfig'))

    for i in range(len(subImageListFiles)):
        subImageListFile = subImageListFiles[i]
        run_str = colmap_exe + ' feature_extractor --ImageReader.single_camera_per_folder 1' + \
            ' --database_path ' + database_path + \
            ' --image_path ' + image_path + \
            ' --image_list_path ' + subImageListFile + \
            ' --ImageReader.camera_model ' + cameraModles[i] + \
            ' --ImageReader.default_focal_length_factor ' + str(focalLenFactors[i])

        if os.path.isdir(mask_path):
            run_str += ' --ImageReader.mask_path ' + mask_path

        print(run_str)
        os.system(run_str)
    
    # clean tmp data
    checkAndDeleteFiles(subImageListFiles)

    return

def run_sequential_matcher(colmap_exe, database_path, overlap = 10):
    # 序列匹配恢复默认参数
    run_str = colmap_exe + ' sequential_matcher ' + \
        ' --database_path ' + database_path + \
        ' --SequentialMatching.overlap ' + str(overlap) +\
        ' --SequentialMatching.quadratic_overlap 1'
    print(run_str)
    os.system(run_str)
    return

def run_custom_matcher(colmap_exe, database_path, match_list_path, min_num_inliers=15):
    run_str = colmap_exe + ' matches_importer ' + \
        ' --database_path ' + database_path + \
        ' --match_list_path ' + match_list_path + \
        ' --SiftMatching.min_num_inliers ' + str(min_num_inliers)

    print(run_str)
    os.system(run_str)
    return

def run_exhaustive_matcher(colmap_exe, database_path, min_num_inliers=15):
    run_str = colmap_exe + ' exhaustive_matcher ' + \
        ' --database_path ' + database_path + \
        ' --SiftMatching.min_num_inliers ' + str(min_num_inliers)

    print(run_str)
    os.system(run_str)
    return

def check_exist_colmap_model(path, format=".bin"):
    flag = False

    cameras_file = os.path.join(path, "cameras" + format)
    images_file = os.path.join(path, "images" + format)
    points_file = os.path.join(path, "points3D" + format)

    if os.path.exists(cameras_file) and os.path.exists(images_file) and os.path.exists(points_file):
        flag = True

    return flag

def move_colmap_model(src_path, dst_path, format=".bin"):
    move_list = []

    src_cameras_file = os.path.join(src_path, "cameras" + format)
    dst_cameras_file = os.path.join(dst_path, "cameras" + format)
    move_list.append([src_cameras_file, dst_cameras_file])

    src_images_file = os.path.join(src_path, "images" + format)
    dst_images_file = os.path.join(dst_path, "images" + format)
    move_list.append([src_images_file, dst_images_file])

    src_points_file = os.path.join(src_path, "points3D" + format)
    dst_points_file = os.path.join(dst_path, "points3D" + format)
    move_list.append([src_points_file, dst_points_file])

    src_project_file = os.path.join(src_path, "project.ini")
    dst_project_file = os.path.join(dst_path, "project.ini")
    move_list.append([src_project_file, dst_project_file])

    for move in move_list:
        if os.path.exists(move[0]):
            shutil.move(move[0], move[1])

    return

def regularize_model_output_path(output_path, format=".bin"):
    flag = check_exist_colmap_model(output_path, format)
    if check_exist_colmap_model(output_path, format):
        model_path = os.path.join(output_path, "0")
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        move_colmap_model(output_path, model_path, format)
    return

def getSubFolders(folderPath):
    subNames = os.listdir(folderPath)

    subFolders = []
    for name in subNames:
        fullFile = os.path.join(folderPath, name)
        if os.path.isdir(fullFile):
            subFolders.append(fullFile)

    return subFolders

def regularizeDirPath(inputPath):
    if os.path.isdir(inputPath) and (not (inputPath[-1] == '/')) and (not (inputPath[-1] == '\\')):
        inputPath += '/'
    return inputPath

def run_mapper(colmap_exe, database_path, image_path, image_list_path, output_path, \
        min_num_matches = 45, init_min_num_inliers = 200, abs_pose_min_num_inliers = 60, mapper_input_model_path="",
        mapper_ba_global_images_ratio=3.0, mapper_ba_global_points_ratio=3.0, mapper_fix_existing_images=0):

    # 清除历史model数据，重新创建model文件夹
    if os.path.exists(output_path) and (not (regularizeDirPath(output_path) in regularizeDirPath(mapper_input_model_path))):
        shutil.rmtree(output_path)

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    run_str = colmap_exe + ' mapper ' + \
        ' --database_path ' + database_path + \
        ' --image_path ' + image_path + \
        ' --image_list_path ' + image_list_path + \
        ' --output_path ' + output_path + \
        ' --Mapper.min_num_matches ' + str(min_num_matches) + \
        ' --Mapper.init_min_num_inliers ' + str(init_min_num_inliers) + \
        ' --Mapper.abs_pose_min_num_inliers ' + str(abs_pose_min_num_inliers) + \
        ' --Mapper.ba_global_images_ratio ' + str(mapper_ba_global_images_ratio) + \
        ' --Mapper.ba_global_points_ratio ' + str(mapper_ba_global_points_ratio) + \
        ' --Mapper.fix_existing_images ' + str(mapper_fix_existing_images)

    if os.path.exists(mapper_input_model_path):
        run_str = run_str + ' --input_path ' + mapper_input_model_path

    print(run_str)
    os.system(run_str)

    # 如果sfm model直接存储在output_path，则将model移动到output_path/0下面
    regularize_model_output_path(output_path, format=".bin")

    # export geos.txt
    sub_folders = getSubFolders(output_path)
    if len(sub_folders) == 1:
        valid_model_path = sub_folders[0]
        colmap_export_geo(valid_model_path, [1,0,0,0,1,0,0,0,1])

    return

def run_point_triangulator(colmap_exe, database_path, image_path, input_path, output_path):
    run_str = colmap_exe + ' point_triangulator ' + \
            ' --database_path ' + database_path + \
            ' --image_path ' + image_path + \
            ' --input_path ' + input_path + \
            ' --output_path ' + output_path
    print(run_str)
    os.system(run_str)
    return

def run_bundle_adjuster(colmap_exe, input_path, output_path):
    run_str = colmap_exe + ' bundle_adjuster ' + \
              ' --input_path ' + input_path + \
              ' --output_path ' + output_path
    print(run_str)
    os.system(run_str)
    return

def run_model_merger(colmap_exe, local_model, map_model, out_model, max_reproj_error=64, min_2d_inlier_percent=0.6827):
    '''
    min_2d_inlier_percent设置为0.6827, 即±1sigma
    这样max_reproj_error就是sigma
    '''
    run_str = colmap_exe + ' model_merger ' + \
              ' --input_path1 ' + map_model + \
              ' --input_path2 ' + local_model + \
              ' --output_path ' + out_model + \
              ' --max_reproj_error ' + str(max_reproj_error) + \
              ' --min_2d_inlier_percent ' + str(min_2d_inlier_percent)

    print(run_str)
    os.system(run_str)

    # 如果sfm model直接存储在output_path，则将model移动到output_path/0下面
    regularize_model_output_path(out_model, format=".bin")

    # export geos.txt
    sub_folders = getSubFolders(out_model)
    if len(sub_folders) == 1:
        valid_model_path = sub_folders[0]
        colmap_export_geo(valid_model_path, [1, 0, 0, 0, 1, 0, 0, 0, 1])

    return

def run_model_sim3_merger(colmap_exe, local_model, map_model, loc2map_sim3_path, out_model):
    if not os.path.isdir(out_model):
        os.mkdir(out_model)
    if loc2map_sim3_path[-1] != '/':
        loc2map_sim3_path = loc2map_sim3_path + '/'
    sim3_path = loc2map_sim3_path + 'sim3.txt'
    if not os.path.isfile(sim3_path):
        print('Error! No file ---> ', sim3_path)
        return

    run_str = colmap_exe + ' model_sim3_merger ' + \
              ' --map_path ' + map_model + \
              ' --loc_path ' + local_model + \
              ' --loc2map_sim3_path ' + loc2map_sim3_path + \
              ' --output_path ' + out_model
    
    print(run_str)
    os.system(run_str)

    return

def run_model_merger_plus(colmap_exe, database_path, image_path, local_model, map_model, out_model, max_reproj_error=64, min_2d_inlier_percent=0.3):
    run_model_merger(colmap_exe, local_model, map_model, out_model, max_reproj_error, min_2d_inlier_percent)
    run_bundle_adjuster(colmap_exe, out_model + '/0', out_model + '/0')
    run_point_triangulator(colmap_exe, database_path, image_path, out_model + '/0', out_model + '/0')
    run_bundle_adjuster(colmap_exe, out_model + '/0', out_model + '/0')
    return

def run_image_registrator(colmap_exe, database_path, input_model_path, output_path, \
        min_num_matches = 45, init_min_num_inliers = 200, abs_pose_min_num_inliers = 60):

    # 清除历史model数据，重新创建model文件夹
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    os.mkdir(output_path)

    if not os.path.exists(input_model_path):
        raise Exception('input_model_path does not exist.')

    run_str = colmap_exe + ' image_registrator ' + \
        ' --database_path ' + database_path + \
        ' --input_path ' + input_model_path + \
        ' --output_path ' + output_path + \
        ' --Mapper.min_num_matches ' + str(min_num_matches) + \
        ' --Mapper.init_min_num_inliers ' + str(init_min_num_inliers) + \
        ' --Mapper.abs_pose_min_num_inliers ' + str(abs_pose_min_num_inliers)

    print(run_str)
    os.system(run_str)

    # 如果sfm model直接存储在output_path，则将model移动到output_path/0下面
    regularize_model_output_path(output_path, format=".bin")

    # export geos.txt
    sub_folders = getSubFolders(output_path)
    if len(sub_folders) == 1:
        valid_model_path = sub_folders[0]
        colmap_export_geo(valid_model_path, [1,0,0,0,1,0,0,0,1])

    return

def update_database_by_model(database_path, output_path):
    if output_path[-1] != '/':
        output_path = output_path + '/'
    model_folders = get_folders_in_folder(output_path)
    # 如果model个数大于1或者为0, 都说明seq-sfm得到的结果不够可信, 这时候就不更新database了
    if len(model_folders) > 1: # 如果文件夹下有多个model
        print("warnning! multi-models in --->", output_path)
        return
    if len(model_folders) == 0:
        print("warnning! no-model in --->", output_path)
        return
    cameras, _, _ = auto_read_model(output_path + model_folders[0])

    db = COLMAPDatabase.connect(database_path)
    cursor = db.cursor()
    for key, cam in cameras.items():
        sql = "UPDATE cameras SET params = ? WHERE camera_id = ?"
        cursor.execute(sql, (array_to_blob(cam.params), key))
    db.commit()
    db.close()
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--colmap_exe', required=True)
    parser.add_argument('--database_path', required=True)
    parser.add_argument('--image_path', required=True)
    parser.add_argument('--image_list_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--seq_match_overlap', type=int, default=10)
    parser.add_argument('--mapper_min_num_matches', type=int, default=45)
    parser.add_argument('--mapper_init_min_num_inliers', type=int, default=200)
    parser.add_argument('--mapper_abs_pose_min_num_inliers', type=int, default=60)
    args = parser.parse_args()
    return args

def run_route_sfm(colmap_exe, database_path, image_path, image_list_path, match_list_path, output_path, \
                matcher_min_num_inliers=15, mapper_min_num_matches=45, \
                mapper_init_min_num_inliers=200, mapper_abs_pose_min_num_inliers=60,
                mapper_input_model_path="", skip_feature_extractor=True,
                mask_path='',
                mapper_fix_existing_images=0):

    if not skip_feature_extractor:
        run_feature_extractor(colmap_exe, database_path, image_path, image_list_path, mask_path=mask_path)

    run_custom_matcher(colmap_exe, database_path, match_list_path, matcher_min_num_inliers)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    run_mapper(colmap_exe, database_path, image_path, image_list_path, output_path,
               mapper_min_num_matches, mapper_init_min_num_inliers, mapper_abs_pose_min_num_inliers,
               mapper_input_model_path=mapper_input_model_path,
               mapper_fix_existing_images=mapper_fix_existing_images)

    # 把sfm重建得到的model的camera参数信息, 更新到database中
    update_database_by_model(database_path, output_path)

    return

def run_model_aligner(colmap_exe, src_model_path, ref_images_path, dst_model_path, max_error=0.05):
    run_str = colmap_exe + ' model_aligner' + \
              ' --input_path ' + src_model_path + \
              ' --ref_images_path ' + ref_images_path + \
              ' --output_path ' + dst_model_path + \
              ' --robust_alignment_max_error ' + str(max_error)

    print(run_str)
    os.system(run_str)
    return

def run_custom_sfm(colmap_exe, database_path, image_path, image_list_path, match_list_path, output_path, \
                matcher_min_num_inliers=15, mapper_min_num_matches=45, \
                mapper_init_min_num_inliers=200, mapper_abs_pose_min_num_inliers=60,
                mapper_input_model_path="",
                skip_feature_extractor=True,
                skip_custom_match=False, 
                apply_model_aligner=False,
                mask_path='',
                mapper_fix_existing_images=0,
                optionalMapper=False,
                use_partial_existing_model=False,
                selected_existing_image_list_path=None):
    
    modelPath = os.path.dirname(output_path)
    _, filename = os.path.split(database_path)
    modelName, _ = os.path.splitext(filename)

    if not skip_feature_extractor:
        run_feature_extractor(colmap_exe, database_path, image_path, image_list_path, mask_path=mask_path)

    if not skip_custom_match:
        run_custom_matcher(colmap_exe, database_path, match_list_path, matcher_min_num_inliers)

    if not optionalMapper:
        if use_partial_existing_model:
            assert os.path.isdir(mapper_input_model_path)
            assert os.path.isfile(selected_existing_image_list_path)

            partial_input_model_path = os.path.join(modelPath, modelName + '_partial_input_model')
            if not os.path.isdir(partial_input_model_path):
                os.mkdir(partial_input_model_path)
            run_image_extractor(colmap_exe, mapper_input_model_path, partial_input_model_path, '', selected_existing_image_list_path)

            partial_output_path = os.path.join(modelPath, modelName + '_partial_output_model')
            run_mapper(colmap_exe, database_path, image_path, image_list_path, partial_output_path,
                    mapper_min_num_matches, mapper_init_min_num_inliers, mapper_abs_pose_min_num_inliers,
                    mapper_input_model_path=partial_input_model_path,
                    mapper_fix_existing_images=mapper_fix_existing_images)

            output_path_leaf = os.path.join(output_path, '0')
            if not os.path.isdir(output_path):
                shutil.rmtree(output_path)
            os.makedirs(output_path_leaf)

            run_model_merger(colmap_exe, os.path.join(partial_output_path, '0'), mapper_input_model_path, output_path)
            colmap_export_geo(output_path_leaf, [1, 0, 0, 0, 1, 0, 0, 0, 1])  
        else:
            run_mapper(colmap_exe, database_path, image_path, image_list_path, output_path,
                    mapper_min_num_matches, mapper_init_min_num_inliers, mapper_abs_pose_min_num_inliers,
                    mapper_input_model_path=mapper_input_model_path,
                    mapper_fix_existing_images=mapper_fix_existing_images)

        # 把sfm重建得到的model的camera参数信息, 更新到database中
        update_database_by_model(database_path, output_path)

        # align output model to mapper_input_model
        output_path_subdirs = getSubFolders(output_path)
        if apply_model_aligner and os.path.exists(mapper_input_model_path) and (len(output_path_subdirs)==1):
            run_model_aligner(colmap_exe, output_path_subdirs[0], os.path.join(mapper_input_model_path, 'geos.txt'),
                              output_path_subdirs[0], max_error=0.05)

            # export geos.txt
            colmap_export_geo(output_path_subdirs[0], [1, 0, 0, 0, 1, 0, 0, 0, 1])

    return

def run_exhaustive_sfm(colmap_exe, database_path, image_path, image_list_path, output_path, \
                matcher_min_num_inliers=15, mapper_min_num_matches=45, \
                mapper_init_min_num_inliers=200, mapper_abs_pose_min_num_inliers=60,
                mapper_input_model_path="", skip_feature_extractor=True, apply_model_aligner=False,
                mask_path='',
                mapper_fix_existing_images=0,
                optionalMapper=False):

    if not skip_feature_extractor:
        run_feature_extractor(colmap_exe, database_path, image_path, image_list_path, mask_path=mask_path)

    run_exhaustive_matcher(colmap_exe, database_path, matcher_min_num_inliers)

    if not optionalMapper:
        run_mapper(colmap_exe, database_path, image_path, image_list_path, output_path,
                   mapper_min_num_matches, mapper_init_min_num_inliers, mapper_abs_pose_min_num_inliers,
                   mapper_input_model_path=mapper_input_model_path,
                   mapper_fix_existing_images=mapper_fix_existing_images)

        # 把sfm重建得到的model的camera参数信息, 更新到database中
        update_database_by_model(database_path, output_path)

        # align output model to mapper_input_model
        output_path_subdirs = getSubFolders(output_path)
        if apply_model_aligner and os.path.exists(mapper_input_model_path) and (len(output_path_subdirs)==1):
            run_model_aligner(colmap_exe, output_path_subdirs[0], os.path.join(mapper_input_model_path, 'geos.txt'),
                              output_path_subdirs[0], max_error=0.05)

            # export geos.txt
            colmap_export_geo(output_path_subdirs[0], [1, 0, 0, 0, 1, 0, 0, 0, 1])

    return

def run_seq_sfm(colmap_exe, database_path, image_path, image_list_path, output_path, \
                seq_match_overlap, mapper_min_num_matches, mapper_init_min_num_inliers, mapper_abs_pose_min_num_inliers,
                mask_path='',
                optionalMapper=False):
    run_database_creator(colmap_exe, database_path)
    run_feature_extractor(colmap_exe, database_path, image_path, image_list_path, mask_path=mask_path)
    run_sequential_matcher(colmap_exe, database_path, seq_match_overlap)

    if not optionalMapper:
        run_mapper(colmap_exe, database_path, image_path, image_list_path, output_path, \
            mapper_min_num_matches, mapper_init_min_num_inliers, mapper_abs_pose_min_num_inliers)
        # 把seq-sfm重建得到的model的camera参数信息, 更新到database中
        update_database_by_model(database_path, output_path)

    return 

def main():
    args = parse_args()
    run_seq_sfm(args.colmap_exe, args.database_path, args.image_path, args.image_list_path, args.output_path, \
                args.seq_match_overlap, args.mapper_min_num_matches, args.mapper_init_min_num_inliers, args.mapper_abs_pose_min_num_inliers)
    return

if __name__ == '__main__':
    main()