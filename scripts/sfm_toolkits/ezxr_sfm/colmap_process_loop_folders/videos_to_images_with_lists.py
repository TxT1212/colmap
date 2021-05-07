# coding: utf-8
import cv2
import os
import argparse
import threading
import time
import sys
from tqdm import tqdm

scriptPath = sys.path[0]
sys.path.append(os.path.dirname(scriptPath))

from fileio.dir_operation import *
from colmap_process.create_file_list import write_image_list
validThread = 5
videoCounter = 0
totalNumber = 0
def getFiles(dir, suffix):
    res = []
    for root, directory, files in os.walk(dir):
        for filename in files:
            name, suf = os.path.splitext(filename)
            for ext in suffix:
                if suf == ext:
                    res.append(os.path.join(root, filename))
    return res

def video2image(video_path, output_path, small_output_path, dstdir, small_dstdir, interval, shortsize, imageListSuffix=None):
    global validThread
    global videoCounter, totalNumber
    validThread = validThread - 1
    print("Open New Thread, Valid_thread = ", validThread)
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = -1
    total_frames = -1
    save_count = 0
    total_frames = vidcap.get(7)
    relative_path = None
    small_relative_path = None
    if dstdir[-1] == '/':
        relative_path = dstdir
    else:
        relative_path = dstdir + '/'
    if small_dstdir[-1] == '/':
        small_relative_path = small_dstdir
    else:
        small_relative_path = small_dstdir + '/'

    print('start video2image ', video_path, ', frames = ', total_frames)
    output_list_path = None
    small_output_list_path = None

    if imageListSuffix == None:
        if output_path[-1] == '/':
            output_list_path = output_path[0:-1] + '_interval_' + str(interval) + '.txt'
        else:
            output_list_path = output_path + '_interval_' + str(interval) + '.txt'
        if small_output_path[-1] == '/':
            small_output_list_path = small_output_path[0:-1] + '_interval_' + str(interval) + '.txt'
        else:
            small_output_list_path = small_output_path + '_interval_' + str(interval) + '.txt'
    else:
        if output_path[-1] == '/':
            output_list_path = output_path[0:-1] + imageListSuffix + '.txt'
        else:
            output_list_path = output_path + imageListSuffix + '.txt'
        if small_output_path[-1] == '/':
            small_output_list_path = small_output_path[0:-1] + imageListSuffix + '.txt'
        else:
            small_output_list_path = small_output_path + imageListSuffix + '.txt'

    image_list = []
    small_image_list = []
    while success:
        count += 1
       # print("{0}, 进度:{1}%\n".format(video_path, round((count + 1) * 100 / total_frames)), end="\r")
        if count%interval == 0:
            image_small = None
            # image.shape存储的是height和width
            if image.shape[0] > image.shape[1]: # 如果height > width
                new_width = int(shortsize)
                new_height = int(image.shape[0] * 1.0 / image.shape[1] * shortsize)
                image_small = cv2.resize(image, (new_width, new_height))
            else: # 如果height <= width
                new_height = int(shortsize)
                new_width = int(image.shape[1] * 1.0 / image.shape[0] * shortsize)
                image_small = cv2.resize(image, (new_width, new_height))
            cv2.imwrite(output_path+"/%08d.jpg" % count, image)#, [int(cv2.IMWRITE_JPEG_QUALITY), 90])     # save frame as jpg file
            cv2.imwrite(small_output_path+"/%08d.jpg" % count, image_small)     # save frame as JPEG file
            image_name = output_path+"/%08d.jpg" % count
            image_list.append(image_name.replace(relative_path, ''))
            small_image_name = small_output_path+"/%08d.jpg" % count
            small_image_list.append(small_image_name.replace(small_relative_path, ''))
            save_count += 1
        
        success,image = vidcap.read()
    vidcap.release()
    cv2.destroyAllWindows()
    print('finish video2image ', video_path)
    print('write image list...')
    write_image_list(output_list_path, image_list)
    write_image_list(small_output_list_path, small_image_list)
    validThread = validThread + 1
    videoCounter += 1
    print("Current Valid_thread = ", validThread)
    print('finished / total =', videoCounter, ' / ', totalNumber)
    return

def extract_image_from_video(srcdir, dstdir, small_dstdir, multithread=True, interval=1, shortsize = 640, imageListSuffix=None):
    if(os.path.isdir(srcdir)):
        if(dstdir.find(srcdir) == 0):
            print("不能将生成的新目录放在源目录下")
            return
        else:
            if not os.path.isdir(dstdir):
                os.mkdir(dstdir)
            srcfilelist, dstfilelist = copytree(srcdir,dstdir)
            if not os.path.isdir(small_dstdir):
                os.mkdir(small_dstdir)
            _, smalldstfilelist = copytree(srcdir,small_dstdir)
           # print(srcfilelist)
    else:
        print("源文件夹不存在")
        return
    
    global videoCounter, totalNumber
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
            print('not valid')
            videoCounter += 1
            continue
        ## 如果已经生成了则跳过
        output_file, suf = os.path.splitext(dstfilelist[index])
        small_output_file, _ = os.path.splitext(smalldstfilelist[index])

        if os.path.exists(output_file):
            print("Jump ", output_file)
            videoCounter += 1
            continue
        if not os.path.isdir(output_file):
            os.mkdir(output_file)
        if os.path.exists(small_output_file):
            print("Jump ", small_output_file)
            videoCounter += 1
            continue
        if not os.path.isdir(small_output_file):
            os.mkdir(small_output_file)
        global validThread
        if multithread:
            while validThread <= 0:
                time.sleep(1)
            t_sing = threading.Thread(target=video2image, args=(srcfilelist[index], output_file, small_output_file, dstdir, small_dstdir, interval, shortsize, imageListSuffix))
            t_sing.start()
        else:
            video2image(srcfilelist[index], output_file, small_output_file, dstdir, small_dstdir, interval, shortsize, imageListSuffix)
    while videoCounter < len(srcfilelist):
        time.sleep(1)
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--srcpath', required=True)
    parser.add_argument('--dstpath', required=True)
    parser.add_argument('--smalldstpath', required=True)
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--shortsize', type=int, default=640)
    parser.add_argument('--multithread', action='store_true')
    parser.add_argument('--imageListSuffix', default=None, type=str)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print('args.srcpath = ', args.srcpath)
    print('args.dstpath = ', args.dstpath)
    print('args.smalldstpath = ', args.smalldstpath)
    print('args.interval = ', args.interval)
    print('args.shortsize = ', args.shortsize)
    print('args.multithread = ', args.multithread)
    print('args.imageListSuffix = ', args.imageListSuffix)
    extract_image_from_video(args.srcpath, args.dstpath, args.smalldstpath, args.multithread, args.interval, args.shortsize, args.imageListSuffix)

if __name__ == '__main__':
    main()
        

