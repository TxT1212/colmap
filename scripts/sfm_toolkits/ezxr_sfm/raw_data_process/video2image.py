# coding: utf-8
import cv2
import os
import argparse
import sys
import threading
import time
from tqdm import tqdm

sys.path.append('../')
from fileio.dir_operation import *

validThread = 5
videoCounter = 0
totalNumber = 0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--srcpath', required=True)
    parser.add_argument('--dstpath', required=True)
  #  parser.add_argument('--video_ext', required=True)
   # parser.add_argument('--output_path', required=True)
    parser.add_argument('--flip180', action='store_true')
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--resize', action='store_true')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--clip', type=int, default=-1)
    parser.add_argument('--multithread', action='store_true')

    args = parser.parse_args()
    return args

def getFiles(dir, suffix):
    res = []
    for root, directory, files in os.walk(dir):
        for filename in files:
            name, suf = os.path.splitext(filename)
            for ext in suffix:
                if suf == ext:
                    res.append(os.path.join(root, filename))
    return res

def video2image(video_path, output_path, interval=1, clip=-1, flip180=False, resizeImage=False, width=1280, height=720):
    print('start video2image ', video_path)
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = -1
    total_frames = -1
    save_count = 0
    sub_folder = 1
    outpath_copy = output_path
    while success:
        total_frames += 1
        success = vidcap.read()

    while success:
        count += 1
        print("进度:{0}%".format(round((count + 1) * 100 / total_frames)), end="\r")
        if count%interval == 0:
            if flip180:
                cv2.flip(image,0,image)
                cv2.flip(image,1,image)
            if resizeImage:
                image = cv2.resize(image, (width, int(width / image.shape[1] * image.shape[0])))
            cv2.imwrite(output_path+"/%08d.jpg" % count, image)     # save frame as JPEG file
            save_count += 1
        if clip>0 and save_count>=clip:
            output_path = outpath_copy+'_'+str(sub_folder)
            save_count = 0
            sub_folder += 1
            if not os.path.exists(output_path):
                os.makedirs(output_path)

        success,image = vidcap.read()
    print('finish video2image ', video_path)
    vidcap.release()
    cv2.destroyAllWindows()
        # print('Read a new frame: ', success)
        

def videofolder2images(video_path, output_path, multithread=True, interval=1, clip=-1, flip180=False, resizeImage=False, width=1280, height=720):
    for vfile in getFiles(video_path, ['.mp4', '.avi', '.MOV','.MP4','.AVI','.mov', 'insv']):
        (filepath,tempfilename) = os.path.split(vfile)
        (shortname,extension) = os.path.splitext(tempfilename)
        output_subpath = output_path + "/" + shortname
        folder = os.path.exists(output_subpath) 
        if not folder:
            os.makedirs(output_subpath) 
        print(output_subpath)

        if multithread:
            t_sing = threading.Thread(target=video2image, args=(vfile, output_subpath, interval, clip, flip180, resizeImage, width, height))
            t_sing.start()
            t_sing.join()
        else:
            video2image(vfile, output_subpath, interval, clip, flip180, resizeImage, width, height)


def create_preview_video(dataset_path, overwrite=False, multithread=True, interval=1, clip=-1, flip180=False, resizeImage=False, width=1280, height=720):
    for root, device_dirs, files in os.walk(dataset_path):
        sorted(device_dirs)
        for dir in device_dirs:
            dir = os.path.join(root, dir)
            print(dir)
        if resizeImage:
            sizename = str(width)
        else:
            sizename = "org"
        
        # export_dir = "inv"+str(interval)+"_"+sizename+"_clip"+str(clip)

        for dir in device_dirs:
            dir = os.path.join(root, dir)
            full_export_dir = os.path.join(dir, export_dir)
            if os.path.exists(full_export_dir) and not overwrite:
                print(full_export_dir, ' exist, skip it')
                continue
            video_dir = os.path.join(dir, 'videos')
            if not os.path.exists(video_dir):
                print('no video found in', full_export_dir, ', skip it')
                continue
            
            videofolder2images(video_dir, full_export_dir, multithread, interval, clip, flip180, resizeImage, width, height)
        break

def video2image(video_path, output_path, interval=1, clip=-1, flip180=False, resizeImage=False, width=1280, height=720):
    global validThread, videoCounter, totalNumber
    validThread = validThread - 1
    print("Open New Thread, Valid_thread = ", validThread)
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = -1
    total_frames = -1
    save_count = 0
    sub_folder = 1
    outpath_copy = output_path
    total_frames = vidcap.get(7)

    print('start video2image ', video_path, ', frames = ', total_frames)
    # vidcap = cv2.VideoCapture(video_path)
    # success,image = vidcap.read()

    while success:
        count += 1
       # print("{0}, 进度:{1}%\n".format(video_path, round((count + 1) * 100 / total_frames)), end="\r")
        if count%interval == 0:
            if flip180:
                cv2.flip(image,0,image)
                cv2.flip(image,1,image)
            if resizeImage:
                image = cv2.resize(image, (width, int(width / image.shape[1] * image.shape[0])))
            cv2.imwrite(output_path+"/%08d.png" % count, image)     # save frame as JPEG file
            save_count += 1
        if clip>0 and save_count>=clip:
            output_path = outpath_copy+'_'+str(sub_folder)
            save_count = 0
            sub_folder += 1
            if not os.path.exists(output_path):
                os.makedirs(output_path)

        success,image = vidcap.read()
    print('finish video2image ', video_path)
    vidcap.release()
    cv2.destroyAllWindows()
    validThread += 1
    videoCounter += 1
    print("Valid_thread = ", validThread)
    print('finished / total =', videoCounter, ' / ', totalNumber)

def extract_image_from_video(srcdir, dstdir, multithread=True, interval=1, clip=-1, flip180=False, resizeImage=False, width=1280, height=720):
    if(os.path.isdir(srcdir)):
        if(dstdir.find(srcdir) == 0):
            print("不能将生成的新目录放在源目录下")
            return
        else:
            if not os.path.isdir(dstdir):
                os.mkdir(dstdir)
            srcfilelist, dstfilelist = copytree(srcdir,dstdir)
           # print(srcfilelist)
    else:
        print("源文件夹不存在")
        return
    global validThread, videoCounter, totalNumber
    videoCounter = 0
    totalNumber = len(srcfilelist)
    valid_video = ['.mp4', '.avi', '.MOV','.MP4','.AVI','.mov', '.insv']
    for index in range(len(srcfilelist)):
        name, suf = os.path.splitext(srcfilelist[index])
        valid = False
        for ext in valid_video:
            if suf == ext:
                valid = True
        if not valid:
            videoCounter += 1
            continue
        ## 如果已经生成了则跳过
        output_file, suf = os.path.splitext(dstfilelist[index])
        if os.path.exists(output_file):
            print("Jump ", output_file)
            videoCounter += 1
            continue
        if not os.path.isdir(output_file):
            os.mkdir(output_file)
        if multithread:
            while validThread <= 0:
                time.sleep(1)
            t_sing = threading.Thread(target=video2image, args=(srcfilelist[index], output_file, interval, clip, flip180, resizeImage, width, height))
            t_sing.start()
        else:
            video2image(srcfilelist[index], output_file, interval, clip, flip180, resizeImage, width, height)
    while videoCounter < len(srcfilelist):
        time.sleep(1)

def main():
    args = parse_args()
    print('args.srcpath = ', args.srcpath)
    print('args.dstpath = ', args.dstpath)
    print('args.multithread = ', args.multithread)
    print('args.interval = ', args.interval)
    print('args.flip180 = ', args.flip180)
    print('args.resize = ', args.resize)
    print('args.width = ', args.width)
    print('args.height = ', args.height)
    extract_image_from_video(args.srcpath, args.dstpath, args.multithread, args.interval, args.clip, args.flip180, args.resize, args.width, args.height)
    


if __name__ == '__main__':
    main()
        

