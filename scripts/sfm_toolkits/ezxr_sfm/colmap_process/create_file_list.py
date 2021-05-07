# coding: utf-8
import argparse
import glob
import sys
import os
sys.path.append('../')
from colmap_process.create_colmap_custom_match_file_list import *

def create_image_list(image_path, folders, image_ext):
  image_list = []
  for folder in folders:
    images = searchDirFile(folder, image_ext)
    for image in images:
      name1 = image.replace(image_path + '/', '')
      image_list.append(name1)

      # example: sort filelist by numbers in filename, e.g. db/1000.jpg  db/1.jpg
      #image_list = sorted(image_list, key=lambda x: int(x[3:-4]))

  return image_list

def write_image_list(filename, image_list):
  with open(filename, "w") as fout:
    for name in image_list:
      fout.write(name + "\n")

def create_all_image_lists_in_one_folder(images_folder):
  '''
  按照SFM建图流水线2.0的设计,目前的文件结构为:
  images
    video1
      000.png
      001.png
    video2
      000.png
      001.png
  '''
  folders = os.listdir(images_folder)
  for folder in folders: # 获取所有videox的文件夹名字
    if not os.path.isdir(images_folder + folder):
      continue
    names = os.listdir(images_folder + folder) # 获取videox文件夹下的所有图像的名字
    image_name_list = []
    for name in names:
      image_name = folder + '/' + name
      image_name_list.append(image_name)
    image_list_path_name = images_folder + folder + '.txt'
    print('create --> ', image_list_path_name)
    image_name_list = sorted(image_name_list)

    image_name_list.sort()
    print(image_name_list)
    write_image_list(image_list_path_name, image_name_list)

def create_image_list_exclude(image_path, folders_exclude, image_ext, output_file):
  image_list = []
  images = searchDirFile(image_path, image_ext)

  for image in images:
    jump_image = False
    for folder in folders_exclude:
      if image.startswith(folder):
        jump_image = True
        break
    if not jump_image:
      name1 = image.replace(image_path + '/', '')
      image_list.append(name1)

  print("write image list to ", output_file)
  image_list.sort()
  print(image_list)
  write_image_list(output_file, image_list)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_folder', required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
  folders = [   \

            "/data/Benchmark/Aachen-Day-Night/images/db"


  ]

  image_list = create_image_list("/data/Benchmark/Aachen-Day-Night/images", folders, ['jpg', 'png'])
  write_image_list("/data/Benchmark/Aachen-Day-Night/images/db_list.txt", image_list)
  # args = parse_args()
  # images_folder = args.images_folder
  # if images_folder [-1] != '/':
  #   images_folder = images_folder + '/'
  # create_all_image_lists_in_one_folder(images_folder)