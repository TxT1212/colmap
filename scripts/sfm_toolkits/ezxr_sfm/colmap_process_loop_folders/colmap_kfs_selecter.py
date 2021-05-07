# coding: utf-8
import cv2
import os
import argparse
import time
import sys

sys.path.append('../')

from colmap_process.create_file_list import write_image_list
from colmap_process.colmap_seq_sfm import run_seq_sfm
from colmap_process.colmap_keyframe_selecter import select_kfs
from colmap_process.colmap_model_analyzer import get_folders_in_folder

from colmap_process_loop_folders.colmap_seq_sfms import run_colmap_seq_sfms
from colmap_process_loop_folders.tree_topology import getSceneTopology, getTopologyCheckTasks, getOnlyNamesFromNodesInfo, identifyCharucoSeqs

def run_colmap_kfs_selecter(small_image_folder, small_model_folder, seq_list, suffix_str = '', 
                            fat_threshold=0.76, thin_threshold=0.74,
                            pt_ratio_threshold=0.25, pt_num_threshold=100):
    print('---> run_colmap_kfs_selecter ...')
    print('small_image_folder = ', small_image_folder)
    print('small_model_folder = ', small_model_folder)

    # 文件路径字符统一
    if small_image_folder[-1] != '/':
        small_image_folder = small_image_folder + '/'
    if small_model_folder[-1] != '/':
        small_model_folder = small_model_folder + '/'
    
    # 现在支持外部传进来的list变量
    model_folder_name_list = None
    if seq_list is not None:
        # 直接把它赋给循环变量
        model_folder_name_list = seq_list
    else:
        # 循环model_folder下的所有文件夹
        model_folder_name_list = get_folders_in_folder(small_model_folder)
    # 主循环
    line_str_list = []
    for folder in model_folder_name_list:
        sub_model_folder = small_model_folder + folder + suffix_str
        if not os.path.isdir(sub_model_folder):
            print("warnning! no folder named --->", sub_model_folder)
            continue
        cur_folders = get_folders_in_folder(sub_model_folder)
        
        kf_img_count = 0
        image_count = 0 # 不读取database，无法知道multi-model有多少张重复的图像
        count_pt_num = 0
        small_kf_list_path = small_image_folder + folder + '_ptratio_' + str(pt_ratio_threshold) +'.txt'
        model_str = ''
        
        if (len(cur_folders) == 0): # 没有model，说明初始化都无法成功，直接复制多份同一个image_list
            print("warnning! NO model in --->", sub_model_folder)
            print('copy original image_list...')
            origin_image_list = small_image_folder + folder + suffix_str + '.txt'
            run_str = 'cp ' + origin_image_list + ' ' + small_kf_list_path
            print(run_str)
            line_str = folder + '<no-model>--->copy original image_list...\n'
            line_str_list.append(line_str)
            os.system(run_str)
            continue
        elif (len(cur_folders) > 1): # 多个model，有可能多个model的图像加起来也不是全部图像，即部分图像纹理特别差，也是直接丢掉吧
            print("warnning! multi-models in --->", sub_model_folder)
            print('select_kfs for each sub-model...')
            model_str = '<multi-models>'
            small_kf_list_path_list = []
            # 对每个model单独做select_kfs
            for cur_folder in cur_folders:
                small_kf_list_path_tmp = small_kf_list_path[0:-4] + '_' + cur_folder + '.txt'
                cur_model_folder = os.path.join(sub_model_folder, cur_folder)
                kf_image_count_tmp, image_count_tmp, count_pt_num_tmp = select_kfs(cur_model_folder, small_kf_list_path_tmp, 
                                                    fat_threshold, thin_threshold, 
                                                    pt_ratio_threshold, pt_num_threshold)
                image_count = image_count + image_count_tmp # 对于multi-model来说，image_count >= 真实的image_count
                count_pt_num += count_pt_num_tmp
                kf_img_count += kf_image_count_tmp
                small_kf_list_path_list.append(small_kf_list_path_tmp)
            # 然后把多个model的临时kf文件合并成一个kf文件
            lines_set = set() # 可能多个子模型有重复图像，用set
            for small_kf_list_path_tmp in small_kf_list_path_list:
                fo = open(small_kf_list_path_tmp, "r")
                for line in fo.readlines():
                    line = line.strip() # write_image_list里面会加'\n'，所以这里必须把已有的'\n'去掉
                    lines_set.add(line)
                fo.close()
            lines_list = sorted(list(lines_set)) # 排个序，好看间隔多少
            write_image_list(small_kf_list_path, lines_list)
            delta_count = kf_img_count - len(lines_list)
            kf_img_count = len(lines_list)
            count_pt_num = count_pt_num - delta_count
        else: # 1个model，有可能不是100%图像注册进去，但是这里假设注册不进去的图像，纹理特别差，所以直接丢掉吧
            model_str = '<1-model>'
            cur_model_folder = sub_model_folder + '/0/'
            kf_img_count, image_count, count_pt_num = select_kfs(cur_model_folder, small_kf_list_path, 
                                                    fat_threshold, thin_threshold, 
                                                    pt_ratio_threshold, pt_num_threshold)
        ratio = 1.0
        pt_ratio = 1.0
        if image_count > 0:
            ratio = float(kf_img_count) / float(image_count)
            pt_ratio = float(count_pt_num) / float(kf_img_count)
        line_str = folder + model_str + '--->kf_img_count/image_count = ' + \
            str(kf_img_count) + '/' + str(image_count) + ' = ' + str(ratio) + \
            '; \npt_num_count/kf_img_count = ' + str(count_pt_num) + '/' + str(kf_img_count) + ' = ' + str(pt_ratio) + '\n'
        line_str_list.append(line_str)
    return line_str_list

def cp_image_lists(small_image_folder, small_model_folder, seq_list, suffix_str, 
                            pt_ratio_charuco, pt_ratio_common):
    print('---> cp_image_lists ...')
    # print('small_image_folder = ', small_image_folder)
    # print('small_model_folder = ', small_model_folder)

    # 文件路径字符统一
    if small_image_folder[-1] != '/':
        small_image_folder = small_image_folder + '/'
    if small_model_folder[-1] != '/':
        small_model_folder = small_model_folder + '/'
    
    # -------------------------------------------------
    # 现在支持外部传进来的list变量
    # -------------------------------------------------
    model_folder_name_list = None
    if seq_list is not None:
        # 直接把它赋给循环变量
        model_folder_name_list = seq_list
    else:
        # 循环model_folder下的所有文件夹
        model_folder_name_list = get_folders_in_folder(small_model_folder)
    # 主循环
    for folder in model_folder_name_list:
        pt_ratio_charuco_list_path = small_image_folder + folder + '_ptratio_' + str(pt_ratio_charuco) +'.txt'
        pt_ratio_common_list_path = small_image_folder + folder + '_ptratio_' + str(pt_ratio_common) +'.txt'
        assert os.path.isfile(pt_ratio_charuco_list_path)
        run_str = 'cp ' + pt_ratio_charuco_list_path + ' ' + pt_ratio_common_list_path
        print(run_str)
        os.system(run_str)
    return

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--projPath', required=True)
    parser.add_argument('--projName', type=str, default=None, help='如果使用了--validSeqsFromJson，则必须指定该参数')
    parser.add_argument('--batchName', type=str, default=None, help='如果使用了--validSeqsFromJson，则必须指定该参数')

    parser.add_argument('--imagesDsDir', default='images_ds')
    parser.add_argument('--dbDsDir', default='database_ds')
    parser.add_argument('--modelDsDir', default='sparse_ds')
    parser.add_argument('--imageListSuffix', type=str, default='', help='请输入videos解析生成的image_list的后缀')
    parser.add_argument('--configDir', default='config')
    parser.add_argument('--validSeqsFromJson', action='store_false')
    parser.add_argument('--sceneGraph', type=str, default='scene_graph')

    parser.add_argument('--colmapPath', default="colmap")
    parser.add_argument('--use_adaptive_threshold', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    imagesDsPath = os.path.join(args.projPath, args.imagesDsDir)
    configPath = os.path.join(args.projPath, args.configDir)

    #---------------------------------------------------------------------------
    # build scene topology and get task lists for topology check
    # 总文件列表的入口，文件IO如果出错，首先debug这里
    if args.validSeqsFromJson:
        sceneTopology = getSceneTopology(args.projPath, configPath=configPath, sceneGraphName=args.sceneGraph,
                                         validSeqsFromJson=args.validSeqsFromJson, imagesDir=args.imagesDsDir)
    else:
        sceneTopology = getSceneTopology(args.projPath, projName=args.projName, batchName=args.batchName,
                                         sceneGraphName=args.sceneGraph, imagesDir=args.imagesDsDir)

    # 这里只需要用到seq的信息
    seqList, routeList, routePairList = getTopologyCheckTasks(sceneTopology)
    only_name_list = getOnlyNamesFromNodesInfo([seqList, routeList, routePairList])
    full_seq_list = only_name_list[0]

    # 区分charuco序列和非charuco序列
    full_seq_list_charuco_flags = identifyCharucoSeqs(full_seq_list, sceneTopology)
    common_seq_list = []
    charuco_seq_list = []
    for seq_str in full_seq_list:
        if not full_seq_list_charuco_flags[seq_str]:
            common_seq_list.append(seq_str)
        else:
            charuco_seq_list.append(seq_str)

    dbDsPath = os.path.join(args.projPath, args.dbDsDir)
    if not os.path.isdir(dbDsPath):
            os.mkdir(dbDsPath)
    modelDsPath = os.path.join(args.projPath, args.modelDsDir)
    if not os.path.isdir(modelDsPath):
            os.mkdir(modelDsPath)
    
    #---------------------------------------------------------------------------
    # 先有model，才能做kf的筛选
    # 参数全都设置成colmap默认参数，这里不需要它重建精度高，要保证成功率
    run_colmap_seq_sfms(args.colmapPath, imagesDsPath, dbDsPath, modelDsPath, full_seq_list, suffix_str=args.imageListSuffix,
                        seq_match_overlap=10, mapper_min_num_matches=15,
                        mapper_init_min_num_inliers=100, mapper_abs_pose_min_num_inliers=30)
    if not args.use_adaptive_threshold:
        #---------------------------------------------------------------------------
        # 分级做kf的筛选，采样率由低到高; 
        report_kf = imagesDsPath + '/report_kf' + args.imageListSuffix + '.txt'
        infile = open(report_kf, 'w')
        # colmap init inlier number默认是100
        common_pt_num_threshold = 100
        # 循环不同采样率
        pt_ratio_threshold_list = [0.3, 0.4, 0.5, 0.6] # ratio最低是0.1，保证在纹理很好的地方多删除图像
        for pt_ratio in pt_ratio_threshold_list:
            line_str = 'pt_ratio = ' + str(pt_ratio) + '\n'
            infile.write(line_str)
            line_str_list = run_colmap_kfs_selecter(imagesDsPath, modelDsPath, common_seq_list, suffix_str=args.imageListSuffix, 
                                    fat_threshold=0.76, thin_threshold=0.74,
                                    pt_ratio_threshold=pt_ratio, pt_num_threshold=common_pt_num_threshold)
            for line_str in line_str_list:
                infile.write(line_str)
        
        # charuco只有一个采样率
        pt_ratio_charuco = 0.8 # charuco的采样率更高，保证尺度的精度
        line_str = 'pt_ratio_charuco = ' + str(pt_ratio_charuco) + '\n'
        infile.write(line_str)
        line_str_list = run_colmap_kfs_selecter(imagesDsPath, modelDsPath, charuco_seq_list, suffix_str=args.imageListSuffix, 
                                fat_threshold=0.76, thin_threshold=0.74,
                                pt_ratio_threshold=pt_ratio_charuco, pt_num_threshold=common_pt_num_threshold)
        for line_str in line_str_list:
            infile.write(line_str)
        infile.close()

        # 把只有一个采样率的charuco的list，按照普通采样率的方式，copy几份，统一后续文件io格式
        for pt_ratio in pt_ratio_threshold_list:
            cp_image_lists(imagesDsPath, modelDsPath, charuco_seq_list, args.imageListSuffix, 
                                    pt_ratio_charuco, pt_ratio)
    else: # 用seq-model自适应的参数，但是给兜底的参数
        #---------------------------------------------------------------------------
        # 分级做kf的筛选，采样率由低到高; 
        report_kf = imagesDsPath + '/report_kf' + args.imageListSuffix + '.txt'
        infile = open(report_kf, 'w')
        # colmap init inlier number默认是100
        common_pt_num_threshold = 100
        common_pt_ratio_threshold = 0.3
        line_str = 'adaptive, min_pt_ratio = ' + str(common_pt_ratio_threshold) + '\n'
        infile.write(line_str)
        line_str_list = run_colmap_kfs_selecter(imagesDsPath, modelDsPath, common_seq_list, suffix_str=args.imageListSuffix, 
                                fat_threshold=0.76, thin_threshold=0.74,
                                pt_ratio_threshold=common_pt_ratio_threshold, pt_num_threshold=common_pt_num_threshold)
        for line_str in line_str_list:
            infile.write(line_str)
        
        # charuco只有一个采样率
        pt_ratio_charuco = 0.8 # charuco的采样率更高，保证尺度的精度
        line_str = 'pt_ratio_charuco = ' + str(pt_ratio_charuco) + '\n'
        infile.write(line_str)
        line_str_list = run_colmap_kfs_selecter(imagesDsPath, modelDsPath, charuco_seq_list, suffix_str=args.imageListSuffix, 
                                fat_threshold=0.76, thin_threshold=0.74,
                                pt_ratio_threshold=pt_ratio_charuco, pt_num_threshold=common_pt_num_threshold)
        for line_str in line_str_list:
            infile.write(line_str)
        infile.close()
        cp_image_lists(imagesDsPath, modelDsPath, charuco_seq_list, args.imageListSuffix, 
                                    pt_ratio_charuco, common_pt_ratio_threshold)

if __name__ == '__main__':
    main()