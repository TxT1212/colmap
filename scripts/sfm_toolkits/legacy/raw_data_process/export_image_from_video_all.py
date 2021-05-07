import cv2
import os
import argparse
import threading

class item:
    def __init__(self):
        self.dataset_path = ''     # 名称
        self.interval = 10
        self.flip180 = False
        self.resize = True
        self.clip = 100
        self.width = 1280
        self.height = 720

args = item() # 定义结构对象

args.dataset_path = "/media/administrator/dataset/oasis/LandmarkAR/C11_mobile_dataset"
args.interval = 10
args.flip180 = False
args.resize = True
args.clip = 1000
args.width = 1280
args.height = 720

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True)
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
    save_count = 0
    sub_folder = 1
    outpath_copy = output_path
    while success:
        count += 1
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
        # print('Read a new frame: ', success)
        

def videofolder2images(video_path, output_path, multithread=True, interval=1, clip=-1, flip180=False, resizeImage=False, width=1280, height=720):
    for vfile in getFiles(video_path, ['.mp4', '.avi', '.MOV','.MP4','.AVI','.mov']):
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
        else:
            video2image(vfile, output_subpath, interval, clip, flip180, resizeImage, width, height)

def parse_video_seq(dataset_path, overwrite=False, multithread=True, interval=1, clip=-1, flip180=False, resizeImage=False, width=1280, height=720):
    for root, device_dirs, files in os.walk(dataset_path):
        sorted(device_dirs);
        for dir in device_dirs:
            dir = os.path.join(root, dir)
            print(dir)
        if resizeImage:
            sizename = str(width)
        else:
            sizename = "org"
        
        export_dir = "inv"+str(interval)+"_"+sizename+"_clip"+str(clip)
        for dir in device_dirs:
            dir = os.path.join(root, dir)
            full_export_dir = os.path.join(dir, export_dir)
            if os.path.exists(full_export_dir) and not overwrite:
                print(full_export_dir, ' exist, skip it')
                continue
            video_dir = dir
            # video_dir = os.path.join(dir, 'videos')
            # if not os.path.exists(video_dir):
            #     print('no video found in', full_export_dir, ', skip it')
            #     continue
            
            videofolder2images(video_dir, full_export_dir, multithread, interval, clip, flip180, resizeImage, width, height)
        break

def main():
    args = parse_args()
    parse_video_seq(args.dataset_path, False, args.multithread, args.interval, args.clip, args.flip180, args.resize, args.width, args.height)
    
    #args = parse_args()
  #  videofolder2images(
    #        args.video_path, args.output_path, args.interval, args.flip180, args.resize, args.width, args.height)

#### TODO
# 1. video切段的长度作为参数外置
# 2. 多文件夹的顺序执行
# 文件结构 
#  Dataset
#   - date_device1
#       -videos
#       -images_inv10_1280_cut2000
#   - date_device2


if __name__ == '__main__':
    main()
        

