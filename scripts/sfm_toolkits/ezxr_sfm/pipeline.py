# coding: utf-8
import json
import os
import re
import sys
import shutil
import time
#from tqdm import tqdm


from fileio.move_file import move_file
from fileio.parser_config_info import *
from colmap_script.reconstruction import set_reconstruction_env, run_reconstruction
from custom_commands import run_custom_command
from external_commands import run_external_command

global_paths = []
cameras = []

def handleLogialRunItems(runStatusDict):
    appendDict = {}
    for command in runStatusDict.keys():
        if command.startswith("logical."):
            logicalDcit = runStatusDict[command]
            globalStatus = logicalDcit['global_status']
            
            for subCommand in logicalDcit.keys():
                if not (subCommand == 'global_status'):
                    appendDict[subCommand] = (logicalDcit[subCommand], 0)[globalStatus == 0]

    runStatusDict.update(appendDict)

    return runStatusDict

def run_pipeline(data):
    global global_paths, cameras
    global_paths = convert_path_var(data['paths'], data['paths'], data['paths'])
    commands = data['commands']
    if 'cameras' in data.keys():
        cameras = data['cameras']

    run = handleLogialRunItems(data['run'])
    set_reconstruction_env(global_paths, cameras)

    print(json.dumps(global_paths, indent = 2))
    print(json.dumps(cameras, indent = 2))
    for comd in commands:
        run_status = run.get(comd)
        if run_status == None:
            print("Can't find command ", comd, " in run list")
            continue
        if run_status == 0:
            print("Jump ", comd)
            continue
        if run_status == 2:
            print("******Check Below Result******", comd)

        print_run_tag(comd, commands[comd]['note'])

    time.sleep(1)
    string = input("check params, commands and status, Press to continue...\n")

    parse_commands(commands, run)

def parse_commands(commands, run):
    for comd in commands:
        command = commands[comd]
        # if command['run'] == False:
        #     print("Jump ", comd)
        #     continue
        
        run_status = run.get(comd)
        if run_status == None:
            print("Can't find command ", comd, " in run list")
            input("Press to continue, and jump it\n")
            continue
        if run_status == 0:
            print("Jump ", comd)
            continue
        print_run_tag(comd, command['note'])
        #print(json.dumps(command, indent = 2))
        if comd.startswith("colmap."):
            run_reconstruction(comd, command)
        if comd.startswith("copy."):
            run_copy_item(command)
        if comd.startswith("custom."):
            run_custom_command(comd, command, global_paths)
        if comd.startswith("external."):
            run_external_command(comd, command, global_paths)
        if run_status == 2:
            time.sleep(1)
            print_run_tag(comd, command['note'])
            str = input("Finish, check result, and press to continue\n")

def run_copy_item(copyitem):
# "copy.baseimage":  {
#       "note": "copy base images to colmap_proj_path",
#       "src": "${video_frame_path}/base",
#       "dst": "${model_proj_path}/images"
#     }
    item_cvt = convert_path_var(copyitem, copyitem, global_paths)
    src_path = item_cvt['src']
    dst_path = item_cvt['dst']
   
    print("copy from ", src_path , " to ", dst_path)
    move_file(src_path, dst_path)

def run_command(command_name, command):
    if command['run'] == False:
        return
    print_run_tag("command_name", command['note'])

    #switch command_name, run command

