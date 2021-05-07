# coding: utf-8
import os
import sys
from python_scale_ezxr.charuco_detection import *
from python_scale_ezxr.charuco_match import *
from python_scale_ezxr.scale_restoration import *
from colmap_script.colmap_proj import ColmapProj
from colmap_process.colmap_model_modify import modify_image_model_binary


def mkdir_charuco(colmap_project_root_folder):
    write_txt_folder = colmap_project_root_folder + '/charucos/'
    if not os.path.isdir(write_txt_folder):
        os.mkdir(write_txt_folder)
    write_txt_2d_folder = colmap_project_root_folder + '/charucos/detection/'
    if not os.path.isdir(write_txt_2d_folder):
        os.mkdir(write_txt_2d_folder)
    write_txt_3d_folder = colmap_project_root_folder + '/charucos/triangulation/'
    if not os.path.isdir(write_txt_3d_folder):
        os.mkdir(write_txt_3d_folder)
    write_txt_match_folder = colmap_project_root_folder + '/charucos/match/'
    if not os.path.isdir(write_txt_match_folder):
        os.mkdir(write_txt_match_folder)
    return write_txt_folder, write_txt_2d_folder, write_txt_3d_folder, write_txt_match_folder

# 查找目录下所有子文件夹（文件夹中只含文件，不含文件夹）
def find_sub_dirs(path):
    subdir = []
    files=os.listdir(path)   #查找路径下的所有的文件夹及文件

    is_subdir = True
    for filee in  files:
        subpath=str(path+'/'+filee)    #使用绝对路径
        if os.path.isdir(subpath):  #判断是文件夹还是文件
            is_subdir = False
            sub = find_sub_dirs(subpath)
            subdir = subdir + sub
    if is_subdir:
        subdir.append(path)
    return subdir    

def run_charuco_colmap_triangulation(colmap_project_root_folder, colmap_image_path, colmap_sparse_model_folder, feature_path, charuco_folder_name):
    ## 构建colmap命令结构体
    if charuco_folder_name[-1] == '/':
        charuco_folder_name = charuco_folder_name[0:-1]
    db_name = charuco_folder_name + '.db'
    match_file = charuco_folder_name + '_match.txt'
    model_path = charuco_folder_name + '_model/'
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    colmap_commands = {}
    database_creator = {  \
        "database_path":db_name    \
    }

    feature_importer = {    \
        "database_path":db_name,    \
        "ImageReader.single_camera_per_folder": 1,  \
        "image_path": colmap_image_path,        \
        "import_path":  feature_path    \
    }

    colmap_commands['database_creator'] = database_creator
    colmap_commands['feature_importer'] = feature_importer
    
    for command_name in colmap_commands:
        command = colmap_commands[command_name]
        param_value_str = ColmapProj.parse_colmap_command(command_name, command)
        print(param_value_str)
        os.system(param_value_str) 
    
    ### reorder model by database
    modify_image_model_binary(colmap_sparse_model_folder, db_name, model_path, '.bin')

    colmap_commands = {}
    matches_importer = {  \
        "database_path":db_name,    \
        "match_list_path":match_file,   \
        "match_type": "raw"     \
    }

    point_triangulator = {    \
        "database_path":db_name,    \
        "image_path": colmap_image_path,        \
        "input_path":  model_path,   \
        "output_path":  model_path,      \
        "Mapper.tri_ignore_two_view_tracks":    0
    }
    colmap_commands['matches_importer'] = matches_importer
    colmap_commands['point_triangulator'] = point_triangulator
    
    for command_name in colmap_commands:
        command = colmap_commands[command_name]
        param_value_str = ColmapProj.parse_colmap_command(command_name, command)
        print(param_value_str)
        os.system(param_value_str) 

    charuco_pts_3d = get_charuco_ids_from_model(db_name, model_path, '.bin')
    return charuco_pts_3d


def get_charuco_points3d(colmap_project_root_folder, colmap_image_path, colmap_sparse_model_folder,
                         colmap_image_list_path, feature_path, charuco_folder_name, colmap_exe='colmap'):
    ## 构建colmap命令结构体
    if charuco_folder_name[-1] == '/':
        charuco_folder_name = charuco_folder_name[0:-1]
    db_name = charuco_folder_name + '.db'
    match_file = charuco_folder_name + '_match.txt'
    model_path = charuco_folder_name + '_model/'
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    colmap_commands = {}
    database_creator = { \
        "database_path": db_name \
        }

    feature_importer = { \
        "database_path": db_name, \
        "ImageReader.single_camera_per_folder": 1, \
        "image_path": colmap_image_path, \
        "import_path": feature_path, \
        "image_list_path": colmap_image_list_path\
        }

    colmap_commands['database_creator'] = database_creator
    colmap_commands['feature_importer'] = feature_importer

    for command_name in colmap_commands:
        command = colmap_commands[command_name]
        param_value_str = ColmapProj.parse_colmap_command(command_name, command, colmap_app_path=colmap_exe)
        print(param_value_str)
        os.system(param_value_str)

        ### reorder model by database
    modify_image_model_binary(colmap_sparse_model_folder, db_name, model_path, '.bin')

    colmap_commands = {}
    matches_importer = { \
        "database_path": db_name, \
        "match_list_path": match_file, \
        "match_type": "raw" \
        }

    point_triangulator = { \
        "database_path": db_name, \
        "image_path": colmap_image_path, \
        "input_path": model_path, \
        "output_path": model_path, \
        "Mapper.tri_ignore_two_view_tracks": 0
    }
    colmap_commands['matches_importer'] = matches_importer
    colmap_commands['point_triangulator'] = point_triangulator

    for command_name in colmap_commands:
        command = colmap_commands[command_name]
        param_value_str = ColmapProj.parse_colmap_command(command_name, command, colmap_app_path=colmap_exe)
        print(param_value_str)
        os.system(param_value_str)

    charuco_pts_3d = get_charuco_ids_from_model(db_name, model_path, '.bin')
    return charuco_pts_3d

def get_charuco_scale(temp_data_root_folder, colmap_image_path, colmap_sparse_model_folder,  board_parameters_path,
                      charuco_seqs_list=[], colmap_exe='colmap', image_list_suffix=''):
    if temp_data_root_folder[-1] == '/':
        temp_data_root_folder = temp_data_root_folder[0:-1]
    if colmap_image_path[-1] == '/':
        colmap_image_path = colmap_image_path[0:-1]

    # 构建文件夹系统,存储尺度恢复的中间变量
    write_txt_folder, write_txt_2d_folder, write_txt_3d_folder, write_txt_match_folder = mkdir_charuco(temp_data_root_folder)

    print('-------charuco detection-------')

    subdirs = []
    colmap_image_list_paths = []
    for seqs in charuco_seqs_list:
        subdirs.append(os.path.join(colmap_image_path, seqs))
        colmap_image_list_paths.append(os.path.join(colmap_image_path, seqs + image_list_suffix+'.txt'))

    # print(subdirs)
    for idx in range(len(subdirs)):
        image_folder_name = subdirs[idx]
        relative_image_folder = image_folder_name[len(colmap_image_path) + 1:]
        full_image_folder = image_folder_name  # image_parent_folder + image_folder_name
        detection_folder = write_txt_2d_folder + relative_image_folder
        if not os.path.isdir(detection_folder):
            os.makedirs(detection_folder)

        ### 检测图像中的2d charuco角点
        num_charuco_image = run_charuco_detection_with_imageList(colmap_image_path, detection_folder,
                                                                 board_parameters_path,
                                                                 colmap_image_list_paths[idx],
                                                                 relative_image_folder)
        if num_charuco_image == 0:
            print("charuco not found in ", relative_image_folder)
            continue
        ### colmap三角化出charuco的点，存储到model
        charuco_pts_3d = get_charuco_points3d(temp_data_root_folder, colmap_image_path,
                                                          colmap_sparse_model_folder, colmap_image_list_paths[idx],
                                                          write_txt_2d_folder, detection_folder, colmap_exe=colmap_exe)

        match_folder = write_txt_match_folder + relative_image_folder
        if not os.path.isdir(match_folder):
            os.makedirs(match_folder)
        charuco_gt_match(board_parameters_path, charuco_pts_3d, match_folder)

    match_folders = find_sub_dirs(write_txt_match_folder)
    scale_opt = compute_scale(match_folders, write_txt_folder + '/report.txt')

    return scale_opt

def run_charuco_scale(colmap_project_root_folder, colmap_image_path, colmap_sparse_model_folder, board_parameters_path, selected_image_path=''):
    if colmap_project_root_folder[-1] == '/':
        colmap_project_root_folder = colmap_project_root_folder[0:-1]
    if colmap_image_path[-1] == '/':
        colmap_image_path = colmap_image_path[0:-1]    

    image_parent_folder = colmap_image_path + '/' + selected_image_path
    if image_parent_folder[-1] == '/':
        image_parent_folder = image_parent_folder[0:-1]
    image_folder_names = os.listdir( image_parent_folder )

    # 构建文件夹系统,存储尺度恢复的中间变量
    write_txt_folder, write_txt_2d_folder, write_txt_3d_folder, write_txt_match_folder = mkdir_charuco(colmap_project_root_folder)

    print('-------charuco detection-------')
    subdirs = find_sub_dirs(image_parent_folder)
   # print(subdirs)

    for image_folder_name in subdirs:
        relative_image_folder = image_folder_name[len(colmap_image_path)+1:]
        full_image_folder = image_folder_name #image_parent_folder + image_folder_name
        detection_folder = write_txt_2d_folder + relative_image_folder
        if not os.path.isdir(detection_folder):
            os.makedirs(detection_folder)

        ### 检测图像中的2d charuco角点
        num_charuco_image = run_charuco_detection(colmap_image_path, detection_folder, board_parameters_path, relative_image_folder)
        if num_charuco_image == 0:
            print("charuco not found in ", relative_image_folder)
            continue
        ### colmap三角化出charuco的点，存储到model
        charuco_pts_3d = run_charuco_colmap_triangulation(colmap_project_root_folder, colmap_image_path, colmap_sparse_model_folder, write_txt_2d_folder, detection_folder)

        match_folder = write_txt_match_folder + relative_image_folder
        if not os.path.isdir(match_folder):
            os.makedirs(match_folder)
        charuco_gt_match(board_parameters_path, charuco_pts_3d, match_folder)

    match_folders = find_sub_dirs(write_txt_match_folder)
    scale_opt = compute_scale(match_folders, write_txt_folder + '/report.txt')
    
    return scale_opt
    



def main():
    if len(sys.argv) != 3:
        print('python run_scale_ezxr.py [colmap_project folder] [colmap_sparse_model folder].')
        return
    board_parameters_path = '/home/netease/ARWorkspace/colmap_ezxr/scripts/sfm_toolkits/ezxr_sfm/python_scale_ezxr/config/board_ChArUco_24x12.yaml'
    colmap_project_root_folder = sys.argv[1]
    colmap_sparse_model_folder = sys.argv[2]

    run_charuco_scale(colmap_project_root_folder, colmap_project_root_folder + '/images/', colmap_sparse_model_folder, board_parameters_path, 'charuco')
        
    return

if __name__ == '__main__':
    main()