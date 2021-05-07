# coding: utf-8
import os
import sys
import shutil
from colmap_script.colmap_proj import ColmapProj
from fileio.parser_config_info import *
from fileio.move_file import move_file
from colmap_process.colmap_export_geo import colmap_export_geo

    
def set_reconstruction_env(paths, cameras):
    ColmapProj.init_colmap_env(paths, cameras)

def convert_path_var_colmap(struct, paths):
    struct = convert_path_var(struct, struct, paths)
    for item in struct['colmap']:
        struct['colmap'][item] = convert_path_var(struct['colmap'][item], struct, paths)

def convert_cam_var_colmap(struct, cameras):
    for item in struct['colmap']:
        for param in struct['colmap'][item]:
            param_value = struct['colmap'][item][param]
            if type(param_value) != type('a'):
                continue
            if param_value.startswith( 'camera.' ):
                status, value = find_member(cameras, param_value[7:])
                if status:
                    struct['colmap'][item][param] = value
                else:
                    print("Error! ", param_value , " not found")
                    sys.exit()

def run_colmap_commands(recons_data):
    recons_proj = ColmapProj(recons_data)
    colmap_commands = recons_data['colmap']
    parse_run_colmap_commands(recons_proj, colmap_commands)
    

def parse_run_colmap_commands(recons_proj, colmap_commands):
    all_command = ''
    if os.path.exists(recons_proj.output_model):
        shutil.rmtree(recons_proj.output_model)
        os.mkdir(recons_proj.output_model) 
    for command_name in colmap_commands:
        command = colmap_commands[command_name]
        param_value_str = recons_proj.parse_colmap_command(command_name, command)
        print(param_value_str)
        all_command = all_command + ' ' + param_value_str + ' && '
    all_command = all_command[:-4]
   # print(all_command)
#s = input("check colmap commands")
    os.system(all_command) 

def run_reconstruction(recons_name, recons_data):
    convert_path_var_colmap(recons_data, ColmapProj.global_paths)
    convert_cam_var_colmap(recons_data, ColmapProj.cameras)
    print(json.dumps(recons_data, indent = 2))

    if recons_name == 'colmap.base_reconstruction':
        base_reconstruction(recons_name, recons_data)

    if recons_name == 'colmap.gravity':
        gravity(recons_name, recons_data)   

    if recons_name == 'colmap.dense_reconstruction' \
    or recons_name == 'colmap.hfnet_database' :
        run_colmap_commands(recons_data)

    if recons_name == 'colmap.image_reg' \
    or recons_name == 'colmap.makegeo' \
    or recons_name == 'colmap.charuco_registration' \
    or recons_name == 'colmap.hfnet_model':
        run_colmap_commands(recons_data)
        colmap_export_geo(recons_data['output_model'], [1,0,0,0,1,0,0,0,1])
    

def base_reconstruction(recons_name, recons_data):
    run_colmap_commands(recons_data)
    output_model = recons_data['output_model']
    ### 检查base_model是否正常完整，且只有一个，如果出现多个，退出
    count = 0
    for fn in os.listdir(output_model): #fn 表示的是文件名
        count = count + 1
    if count != 1:
        print("Warning! base reconstruction produced multiple models or failed with model num = ", count, ". exit and check manually.")
        sys.exit()
    
    move_file(output_model + '/0', output_model, False)
    os.rmdir(output_model + '/0')

    ### create geos.txt
    colmap_export_geo(output_model, [1,0,0,0,1,0,0,0,1])

def gravity(recons_name, recons_data):
    output_model = recons_data['output_model']
    if os.path.exists(recons_data['gravity_model']):
        shutil.rmtree(recons_data['gravity_model'])
    os.mkdir(recons_data['gravity_model'])
    recons_proj = ColmapProj(recons_data)
    colmap_commands = recons_data['colmap']
    run_colmap_commands(recons_data)

    colmap_export_geo(output_model, [1,0,0,0,0,1,0,-1,0])
    #colmap_export_geo(output_model, [1,0,0,0,1,0,0,0,1])
