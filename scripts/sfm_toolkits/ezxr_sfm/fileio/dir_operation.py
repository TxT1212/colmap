# coding: utf-8
# 功能文件，无外接脚本
import os
import shutil
from shutil import *
import sys

# 查找目录下所有子文件夹（文件夹中只含文件，不含文件夹）
def find_sub_dirs(path):
    subdir = []
    files=os.listdir(path)   #查找路径下的所有的文件夹及文件

    is_subdir = True
    for filee in  files:
        subpath=str(path+'/'+filee)    #使用绝对路径
        if os.path.isdir(subpath):  #判断是文件夹还是文件
            is_subdir = False
            sub = find_sub_dirs(subpath)
            subdir = subdir + sub
    if is_subdir:
        subdir.append(path)
    return subdir    

def copytree(src, dst, copyfile=False, symlinks=False, ignore=None):
    names = os.listdir(src)
    srcfilelist = []
    dstfilelist = []
    if ignore is not None:
        ignored_names = ignore(src, names)
    else:
        ignored_names = set()

    if not os.path.isdir(dst): # This one line does the trick
        os.makedirs(dst)
    errors = []
    for name in names:
        if name in ignored_names:
            continue
        srcname = os.path.join(src, name)
        dstname = os.path.join(dst, name)
        try:
            if symlinks and os.path.islink(srcname):
                linkto = os.readlink(srcname)
                os.symlink(linkto, dstname)
            elif os.path.isdir(srcname):
                srcfiles, dstfiles = copytree(srcname, dstname, symlinks, ignore)
                srcfilelist = srcfilelist + srcfiles
                dstfilelist = dstfilelist + dstfiles
            else:
                # Will raise a SpecialFileError for unsupported file types
                if copyfile:
                    copy2(srcname, dstname)
                srcfilelist.append(srcname)
                dstfilelist.append(dstname)
                #copy2(srcname, dstname)
        # catch the Error from the recursive copytree so that we can
        # continue with other files
        except Error as err:
            errors.extend(err.args[0])
        except EnvironmentError as why:
            errors.append((srcname, dstname, str(why)))
    # try:
    #     copystat(src, dst)
    # except OSError as why:
    #     if WindowsError is not None and isinstance(why, WindowsError):
    #         # Copying file access times may fail on Windows
    #         pass
    #     else:
    #         errors.extend((src, dst, str(why)))
    # if errors:
    #     raise errors

    return srcfilelist, dstfilelist

if __name__=="__main__":
    source = os.path.realpath(sys.argv[1]) 
    target = os.path.realpath(sys.argv[2])
    if(os.path.isdir(source)):
        if(target.find(source) == 0):
            print("不能将生成的新目录放在源目录下")
        else:
            if not os.path.isdir(target):
                os.mkdir(target)
            srcfilelist, dstfilelist = copytree(source,target)
            print(srcfilelist)
    else:
        print("源文件夹不存在")
    