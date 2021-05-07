import cv2
import os
import argparse
import threading
import shutil


image_database_path = '/media/administrator/dataset/oasis/LandmarkAR/C11_mobile_dataset/'

colmap_project_path = '/media/administrator/dataset/oasis/LandmarkAR/C11_base'

overwrite = False

def isImageFolder(dir):
    hasImage = False
    for file in os.listdir(dir):
        if file.endswith('jpg'):
            hasImage = True
    return hasImage

for root, device_dirs, files in os.walk(image_database_path):
    device_dirs=sorted(device_dirs)
    for dir in device_dirs:
        dir = os.path.join(root, dir)
        print(dir)
       
        for root_, dirs, files in os.walk(dir):
            print('root=',root_)
            if (len(dirs)<2):
                print('There is no image folder in this path, please run run_video2image.sh first !')
                exit(1);
            dirs=sorted(dirs);

            for dir in dirs:
                print('dir=',dir)
                ifvideopath = dir
                if (ifvideopath == 'videos'):    #跳过videos这个文件夹
                    continue
                dir = os.path.join(root_, dir)
                ### 这里应该直接写到存放image的子路径  不然路径中也可能会有video字段出现，而且一个文件夹内可能存在多份video2image的文件夹
                # if 'video' not in dir:
                for subroot, subdirs, subfiles in os.walk(dir):
                    if (len(subdirs)<1):
                        break;
                    for curr_dir in subdirs:
                        prefix = curr_dir
                        if (prefix[0] == '.' or '_model' in prefix):   #跳过以._开头的文件夹，跳过带有‘_model’的文件夹
                            continue
                        curr_dir = os.path.join(subroot, curr_dir)
                        #print('debug curr_dir: ', curr_dir)
                        #prefix = 'IMG_3303_4'
                        #curr_dir = '/media/administrator/dataset/oasis/LandmarkAR/C11_mobile_dataset/C11_03012_iphonex_vertical_cloudy_auto/inv10_1280_clip1000/IMG_3303_4'
                        dest_model = subroot+'/'+prefix+'_model'
                        dest_db = subroot+'/'+prefix+'.db'
                        if os.path.exists(dest_model) and not overwrite:
                            print('model exist, skip it: ', dest_model)
                            continue

                        # if not isImageFolder(curr_dir):
                        #     print('ignore non image folder: ', curr_dir)
                        #     continue
                        #debug the tmp error here

                        print('start imgreg: ', curr_dir)
                        
                        target_name = colmap_project_path+'/images/'+prefix
                        if os.path.exists(target_name):
                            shutil.rmtree(target_name)
                        shutil.copytree(curr_dir, target_name)
                        
                        ifsuccess = os.system("sh run_extra_imgreg.sh " + colmap_project_path)
                        if (ifsuccess>>8!=1):
                            print('error in runing curr_dir !')
                        model_name = colmap_project_path+'/sparse/imagereg'
                        re_name = colmap_project_path+'/sparse/' + prefix+'_model'
                        db_name = colmap_project_path+'/imagereg.db'
                        if os.path.exists(re_name):
                            shutil.rmtree(re_name)
                        #os.rename(colmap_project_path+'/'+prefix, re_name)
                        os.rename(model_name, re_name)
                        
                        if os.path.exists(dest_model):
                            shutil.rmtree(dest_model)
                        if os.path.exists(dest_db):
                            os.remove(dest_db)
                        shutil.copytree(re_name, dest_model)
                        shutil.copy(db_name, dest_db)
                        shutil.rmtree(re_name)
                        shutil.rmtree(target_name)
                        
                        print('finished imgreg: ', curr_dir)
                    break
            break
        
    break