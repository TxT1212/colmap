# coding: utf-8
import os
import sys
import glob
from pathlib import Path


image_ext = ['.JPG', '.jpg', '.jpeg','.JPEG','.png','.bmp']

def create_match_file_list(image_path, match_folder_file, match_list_file):
  with open(match_list_file, "w") as fout:
        # fid.write(HEADER)
        # for _, cam in cameras.items():
        #     to_write = [cam.id, cam.model, cam.width, cam.height, *cam.params]
        #     line = " ".join([str(elem) for elem in to_write])
        #     fid.write(line + "\n")

    with open(match_folder_file, "r") as fid:
      while True:
        line = fid.readline()
        if not line:
          break
        line = line.strip()  
        if len(line) > 0 and line[0] != "#":
          elems = line.split()
          folder1 = elems[0]
          folder2 = elems[1]

          images1 = searchDirFile(image_path + '/' + folder1, image_ext)
          images2 = searchDirFile(image_path + '/' + folder2, image_ext)

          for image1 in images1:
            name1 = image1.replace(image_path + '/', '')
            for image2 in images2:
              name2 = image2.replace(image_path + '/', '')
              fout.write(name1 + ' ' + name2 + "\n")

def create_match_pair(image_path, folder1, folder2, image_ext):
  image_pairs = []
  images1 = searchDirFile(folder1, image_ext)
  images2 = searchDirFile(folder2, image_ext)
  for image1 in images1:
    name1 = image1.replace(image_path + '/', '')
    for image2 in images2:
      name2 = image2.replace(image_path + '/', '')
      image_pairs.append([name1, name2])
  return image_pairs
    

def searchDirFile(rootDir, suffix):
    res = []
    for dir_or_file in os.listdir(rootDir):
        filePath = os.path.join(rootDir, dir_or_file)
        # 判断是否为文件
        if os.path.isfile(filePath):
            # 如果是文件再判断是否以.jpg结尾，不是则跳过本次循环
            for ext in suffix:
              if os.path.basename(filePath).endswith(ext):
                res.append(os.path.join(rootDir, os.path.basename(filePath)))
              else:
                continue
        # 如果是个dir，则再次调用此函数，传入当前目录，递归处理。
        elif os.path.isdir(filePath):
            subres = searchDirFile(filePath, suffix)
            res = res + subres
        else:print('not file and dir ' + os.path.basename(filePath))
    return res

def getFiles(dir, suffix):
    res = []
    for root, directory, files in os.walk(dir):
        for filename in files:
            name, suf = os.path.splitext(filename)
            for ext in suffix:
                if suf == ext:
                    res.append(os.path.join(root, filename))
    return res

def main():
  res = searchDirFile("/media/netease/Storage/LargeScene/Scene/XixiWetland/colmap_model/xray_test_material/images", ["jpg", "png"])
  print(res)
  # create_match_file_list(
  # sys.argv[1], 
  # sys.argv[2],
  # sys.argv[3],
  # )

if __name__ == "__main__":
    main()

