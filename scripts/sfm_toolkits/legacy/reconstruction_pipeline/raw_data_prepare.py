
import cv2
import os
import argparse
import threading
import numpy



### 将视频素材按需求放置到某一路径

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', required=True)
    parser.add_argument('--colmap_path', required=True)
    parser.add_argument('--script_path', required=True)
    parser.add_argument('--video_interval', type=int, default=30)
    parser.add_argument('--video_resize', type=int, default=0)
    parser.add_argument('--video_width', type=int, default=1280)
    parser.add_argument('--video_height', type=int, default=720)
    parser.add_argument('--video_multithread', type=int, default=1)

    # 检查和逐步跳过
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--jump', type=str, nargs='+', default=[])
    args = parser.parse_args()
    return args

def main():    
    args = parse_args()
    check_scripts = args.check

    video_path = os.path.normpath(args.video_path)
    video_frame_path = video_path + '_image'
    print('==================================\nextract frames from video\n==================================\n')
    if 'video' not in args.jump:
        script_dir = args.script_path + "/raw_data_process/"

        full_command_str = 'python ' + script_dir + 'video2image.py'  \
                    + " --srcpath " + video_path + "/"  \
                    + " --dstpath " + video_frame_path + "/"  \
                    + " --interval " + str(args.video_interval) 

        if args.video_multithread:
            full_command_str = full_command_str + " --multithread "

        if args.video_resize:
            full_command_str = full_command_str \
            + " --resize " \
            + " --width " + str(args.video_width) \
            + " --height " + str(args.video_height) 

        print(full_command_str)
        os.system(full_command_str)  
        
        print("finish ", full_command_str, "\npress to continue")
        if check_scripts:
            chartmp = input("")     

    print('==================================\nprepare colmap project\n==================================\n')
    colmap_path = args.colmap_path
    if not os.path.isdir(colmap_path):
        os.mkdir(colmap_path)

    image_path = colmap_path + "/images/"
    sparse_path = colmap_path + "/sparse/"
    if not os.path.isdir(image_path):
        os.mkdir(image_path)
    if not os.path.isdir(sparse_path):
        os.mkdir(sparse_path)

    command = "mv "

    ##剪切粘贴骨架地图到colmap_path
    full_command_str = command + video_frame_path + "/base " + image_path
    print(full_command_str)
    os.system(full_command_str) 

    print('==================================\nreconstruct base model\n==================================\n')
    

if __name__ == '__main__':
    main()