# coding: utf-8
import os
import sys
import shutil
from raw_data_process.video2image import extract_image_from_video
from python_scale_ezxr.run_scale_ezxr import run_charuco_scale
from fileio.parser_config_info import convert_path_var
from colmap_process.colmap_model_rescale import rescale_model
from colmap_process.create_colmap_custom_match_file_list import create_match_pair
from colmap_process.create_file_list import *
from colmap_process.colmap_export_geo import *
from fileio.dir_operation import find_sub_dirs
from colmap_process.colmap_model_modify import modify_image_model_binary

print_command = True
run_command = True

# 自定义命令，可按需自行添加模块。
def run_custom_command(name, command, paths):
    if name == 'custom.rawdata_process':
        rawdata_process(command, paths['video_path'], paths['video_frame_path'])
    elif name == 'custom.create_recons_proj':
        create_recons_proj(paths['model_proj_path'], paths['model_proj_image_path'])
    elif name == 'custom.charuco_detect':
        params_cvt = convert_path_var(command['params'], command, paths)
        command['params'] = params_cvt
        charuco_detect(command, paths['model_proj_path'], paths['model_proj_image_path'])
    elif name == 'custom.rmcharuco':
        command_cvt = convert_path_var(command, command, paths)
        path = command_cvt['src']
        print("rm ", path)
        if os.path.exists(path):
            shutil.rmtree(path)
    elif name == 'custom.charuco_match_list':
        command_cvt = convert_path_var(command, command, paths)
        create_charuco_match_list(command_cvt, paths['model_proj_image_path'])
    elif name == 'custom.create_gravity_list':
        command_cvt = convert_path_var(command, command, paths)
        create_gravity_list(command_cvt)
    elif name == 'custom.model_reorder':
        command_cvt = convert_path_var(command, command, paths)
        params_cvt = convert_path_var(command_cvt['params'], command_cvt, paths)
        command_cvt['params'] = params_cvt
        model_reorder(command_cvt)
    elif name == 'custom.rescale':
        command_cvt = convert_path_var(command, command, paths)
        rescale(command_cvt)
        
    else:
        print("Undefined command: ", name, ", Exit and check.")
        sys.exit()
        

def rawdata_process(rawdata, video_path, video_frame_path):
    params = rawdata['params']
    if print_command:
        print("-----------command note-----------")
        print('extract_image_from_video.py ', video_path, ' ', video_frame_path,                          \
            ' ', params['multithread'], ' ', params['interval'], ' ', params['clip'], ' ', params['flip180'],    \
                ' ', params['resize'], ' ', params['width'], ' ', params['height'])
        print("----------------------------------")
    if not run_command:
        return

    if video_path == video_frame_path:
        print("error! video_frame_path can't be video_path itself.")
        exit
    extract_image_from_video(video_path, video_frame_path,                          \
        params['multithread'], params['interval'], params['clip'], params['flip180'],    \
            params['resize'], params['width'], params['height'])


def create_recons_proj(model_proj_path, model_proj_image_path):
    if print_command:
        print("----------------------------------")
        print('create_recons_proj ', model_proj_path, ' ', model_proj_image_path)
        print("==================================")
    if not run_command:
        return

    colmap_path = model_proj_path
    if not os.path.isdir(colmap_path):
        os.makedirs(colmap_path)

    image_path = model_proj_image_path
    sparse_path = colmap_path + "/sparse/"

    if not os.path.isdir(sparse_path):
        os.mkdir(sparse_path)
    if not os.path.isdir(image_path):
        os.mkdir(image_path)

def charuco_detect(command, model_proj_path, model_proj_image_path):
    params = command['params']
    scale = run_charuco_scale(params['model_proj_path'], params['model_proj_image_path'], params['input_model'], params['board_parameters_path'], params['selected_image_path'])

    if scale == -1:
        print("Error! can't find charuco board, exit")
        sys.exit()

    rescale_model(scale, params['input_model'], params['output_model'])

def create_charuco_match_list(command, model_proj_image_path):

    charuco_folder = command['charuco_folder']
    if not os.path.isdir(charuco_folder):
        os.makedirs(charuco_folder)

    src_folders = command['src_folders']
    dst_folder = command['dst_folder']
    match_list_file = command['match_file']

    subdirs = find_sub_dirs(src_folders)

    image_pairs = []
    for subdir in subdirs:
        pairs = create_match_pair(model_proj_image_path, subdir, dst_folder, ['.jpg', '.png'])
        image_pairs = image_pairs + pairs
    for subdir in subdirs:
        pairs = create_match_pair(model_proj_image_path, subdir, subdir, ['.jpg', '.png'])
        image_pairs = image_pairs + pairs

    with open(match_list_file, "w") as fout:
        for pair in image_pairs:
            fout.write(pair[0] + ' ' + pair[1] + "\n")

def create_gravity_list(command):
    create_image_list_exclude(command['image_path'], [command['folder']], ['.jpg', '.png'], command['gravity_list'])
    #(image_path, folders_exclude, image_ext, output_file):

def model_reorder(command):
    params = command['params']
    modify_image_model_binary(params['input_model'], params['database'], params['output_model'], params['model_ext'])

def rescale(command):
    rescale_model(command['scale'], command['input_model'], command['output_model'])
    colmap_export_geo(command['output_model'], [1,0,0,0,1,0,0,0,1])