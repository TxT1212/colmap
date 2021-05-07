# -*- coding: utf-8 -*
import os
import shutil

def move_file(orgin_path, moved_path, copy = True, overwrite = False):
    if os.path.isfile(orgin_path): 
        if copy:
            shutil.copy(orgin_path, moved_path)
        else:
            shutil.move(orgin_path, moved_path)
        return

    if not os.path.exists(moved_path):
        os.makedirs(moved_path)
    dir_files=os.listdir(orgin_path)#得到该文件夹下所有的文件
    for file in  dir_files:
        file_path=os.path.join(orgin_path,file)   #路径拼接成绝对路径
        if os.path.isfile(file_path): #如果是文件，就打印这个文件路径
            moved_file = os.path.join(moved_path,file)
            if os.path.exists(moved_file):
                if overwrite == False:
                    print(moved_file + " jump" )
                    continue
                else:
                    os.remove(moved_file)
            else:
                if copy:
                    shutil.copyfile(file_path, moved_file)
                else:
                    shutil.move(file_path, moved_path)

        if os.path.isdir(file_path):  #如果目录，就递归子目录
            childMPath = os.path.join(moved_path,file)
            if not os.path.isdir(childMPath):
                os.mkdir(childMPath)
            move_file(file_path,childMPath, copy, overwrite)
