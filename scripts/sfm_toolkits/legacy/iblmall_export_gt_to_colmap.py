import glob
import os
import numpy as np
import math
import argparse

# fx 0 cx
# 0 fy cy
# 0 0 1
# 0 0 0
# R'    这里是TWC
# cop
# w h

# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
# Number of images: 2, mean observations per image: 2
# 1 0.851773 0.0165051 0.503764 -0.142941 -0.737434 1.02973 3.74354 1 P1180141.JPG

# # Camera list with one line of data per camera:
# #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
# # Number of cameras: 3
# 1 SIMPLE_PINHOLE 3072 2304 2559.81 1536 1152
# 2 PINHOLE 3072 2304 2560.56 2560.56 1536 1152
# 3 SIMPLE_RADIAL 3072 2304 2559.69 1536 1152 -0.0218531


def to_tum(rmat, tmat, inverse):
    if inverse == False:
        qua = quaternion_from_matrix(rmat)
        res = np.array([
                    qua[0], qua[1], qua[2], qua[3], 
                    tmat[0, 0], tmat[1,0], tmat[2,0]
                    ])
    else:
        rmat = rmat.T
        tnew = -rmat @ tmat
        qua = quaternion_from_matrix(rmat)
        res = np.array(
            [qua[0], qua[1], qua[2], qua[3], 
                    tnew[0, 0], tnew[1,0], tnew[2,0]
                    ])
    return res

def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q

def searchDirFile(rootDir, prefixroot, ext, filelist):
    for dir_or_file in os.listdir(rootDir):
        filePath = os.path.join(rootDir, dir_or_file)
        if os.path.isfile(filePath):
            if os.path.basename(filePath).endswith(ext):
                #print('imgBox fileName is '+ os.path.basename(filePath))
                #print('add file ', prefixroot  + os.path.basename(filePath))
                filelist.append(prefixroot + os.path.basename(filePath))
            else:
                continue

        elif os.path.isdir(filePath):
            searchDirFile(filePath, prefixroot + dir_or_file + '/', ext, filelist)
        else:print('not file and dir ' + os.path.basename(filePath))

def export_gt_to_colmap(path, output_image, output_camera, is_crop_resize, src_size, crop_size, resize_scale):
    filelist = [path+'/'+ x for x in os.listdir(path)]
    
    filelist = []
    searchDirFile(path, '', '.camera', filelist)

    image_output = open(output_image, 'a+')
    camera_output = open(output_camera, 'a+')


    filenum = 1
    # for idx, file in enumerate(filelist):
    for file in filelist:
        f = open(path+'/'+file, 'r')
        data = f.readlines()
        linenum = 0

        rmat = np.zeros(shape=(3,3))
        tmat = np.zeros(shape=(3,1))


        for line in data:
            odom = line.split()        #将单个数据分隔开存好
            # numbers_float = map(float, odom) #转化为浮点数
            
            # 内参矩阵
            if linenum == 0:
                fx = float(odom[0])
                cx = float(odom[2])

                if is_crop_resize is True:
                    cx = cx - (src_size[0]-crop_size[0]) * 0.5  # crop for cx
                    fx *= resize_scale  # resize for fx
                    cx = cx * resize_scale - 0.5 * resize_scale + 0.5  # resize for cx    

            if linenum == 1:
                fy = float(odom[1])
                cy = float(odom[2]) 

                if is_crop_resize is True:
                    cy = cy - (src_size[1]-crop_size[1]) * 0.5  # crop for cy
                    fy *= resize_scale  # resize for fy
                    cy = cy * resize_scale - 0.5 * resize_scale + 0.5  # resize for cy

            # Rotation
            if linenum >= 4 and linenum <= 6:
                rmat[linenum - 4, 0] = float(odom[0])
                rmat[linenum - 4, 1] = float(odom[1])
                rmat[linenum - 4, 2] = float(odom[2])
            
            # Translation
            if linenum == 7:
                tmat[0, 0] = float(odom[0])
                tmat[1, 0] = float(odom[1])
                tmat[2, 0] = float(odom[2])
            # image size
            if linenum == 8:
                w = float(odom[0])
                h = float(odom[1])
            linenum+=1

        f.close()

        pose = to_tum(rmat, tmat, True)

        imagestr = os.path.splitext(file)[0] + ".jpg"
        linestring = str(filenum) + " " + str(pose[0]) + " " + str(pose[1]) + " " + str(pose[2]) + " " + str(pose[3]) + " " + str(pose[4]) + " " + str(pose[5]) + " " + str(pose[6])  + " " + str(filenum) + " " + imagestr + "\n\n"
        image_output.write(linestring)

        camstring = str(filenum) + ' PINHOLE ' + str(w) + " " + str(h) + " " + str(fx) + " " + str(fy) + " " + str(cx) + " " + str(cy) + "\n"
        camera_output.write(camstring)

        filenum+=1

    image_output.close()
    camera_output.close()
            # for eachLine in f1:
            #     f2.write(eachLine)
            #     f2.write(' '+str(idx+1)+'\n')
            # f1.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)  # ibl数据集根目录
    parser.add_argument('--output_image', required=True)  # ibl数据集根目录下存放*.txt的位置
    parser.add_argument('--output_camera', required=True)
    parser.add_argument('--is_crop_resize', default=False)  # True, if needed, first crop, then resize
    parser.add_argument('--src_size_w', type=int, required=True)  # (w, h), (2992, 2000)
    parser.add_argument('--src_size_h', type=int, required=True)
    parser.add_argument('--crop_size_w', type=int, required=True)  # (w, h), (2992, 1683)
    parser.add_argument('--crop_size_h', type=int, required=True)
    parser.add_argument('--resize_scale', type=float, default=1.0)  # 1.0
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    args.output_image = os.path.join(args.path, args.output_image)
    args.output_camera = os.path.join(args.path, args.output_camera)
    src_size = (args.src_size_w, args.src_size_h)
    crop_size = (args.crop_size_w, args.crop_size_h)
    export_gt_to_colmap(args.path, args.output_image, args.output_camera, args.is_crop_resize, src_size, crop_size, args.resize_scale)

if __name__ == "__main__":
    main()
