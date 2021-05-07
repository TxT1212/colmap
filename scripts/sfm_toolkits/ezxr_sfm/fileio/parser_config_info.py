# -*- coding: utf-8 -*
import json
import os
import shutil
import re
import sys
from collections import OrderedDict
def find_member(struct, target):
    find = False
    if target in struct:
        return True, struct[target]
    return False, 0
    
def print_run_tag(name, note):
    print('==================================\n',name,'\n', note,'\n==================================')

def parse_paths(paths):
    video_path = paths['video_path']
    video_frame_path = paths['video_frame_path']
    colmap_app_path = paths['colmap_app_path']
    model_proj_path = paths['model_proj_path']
    model_image_path = model_proj_path + paths['model_proj_image_path']

    if video_path == video_frame_path:
        print("error! video_frame_path can't be video_path itself.")
        exit
    return video_path, video_frame_path, colmap_app_path, model_proj_path, model_image_path

def parse_config_file(json_file):
    with open(json_file,'r', encoding='UTF-8') as f:
        data = json.load(f, object_pairs_hook=OrderedDict)
        return data

# 当前json单元为local_item，全局path为global_item
def parse_path_var_legacy(paramstr, local_item, global_item):
    matchObj = re.match( r'\${(.*)}(.*)', paramstr, re.M|re.I)
    if matchObj:
        path = matchObj.group(1)
        if path.startswith("global."):
            status, value = find_member(global_item, path[7:])
        else:
            status, value = find_member(local_item, path)
        if status:
            return value +matchObj.group(2)
        else:
            print("Error! ", paramstr , " not found")
            sys.exit()
    else:   
        return paramstr

def parse_path_var(paramstr, local_item, global_item):
    paramstrParts = paramstr.split('$')
    paramstrParsed = ''

    for paramPart in paramstrParts:
        matchObj = re.match('\{(.*)\}(.*)', paramPart, re.M|re.I)

        if matchObj:
            path = matchObj.group(1)
            if path.startswith("global."):
                status, value = find_member(global_item, path[7:])
            else:
                status, value = find_member(local_item, path)

            if status:
                paramPart =  value + matchObj.group(2)
            else:
                print("Error! ", paramstr , " not found")
                sys.exit()

        paramstrParsed += paramPart

    return paramstrParsed

# 将local_item中的${path}格式转换为正常路径
def convert_path_var(this_item, local_item, global_item):
    item_cvt = this_item
    for item in item_cvt:
        if type(item_cvt[item]) == type('a'):
            param_str = parse_path_var(item_cvt[item], local_item, global_item)
            item_cvt[item] = param_str
    return item_cvt

def convert_path_list(this_item, local_item, global_item):
    item_cvt = []
    for item in this_item:
        if type(item) == type('a'):
            param_str = parse_path_var(item, local_item, global_item)
            item = param_str
        item_cvt.append(item)
    return item_cvt