# coding: utf-8
import numpy as np
import sys
import cv2
from cv2 import aruco
import glob
import os
import yaml
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from colmap_process.colmap_read_write_model import *
from colmap_process.colmap_db_parser import *
from python_scale_ezxr.io_tool import *


def create_charuco_corners(board_rows, board_cols, square_length, marker_length):
    """
    key是id, value是物理坐标
    以左下角为原点,横x,竖y,上z,符合右手坐标系
    用row = 4, col = 3举例:
    y
    ^
    |
    |
    x   x   x   x
    x   4   5   x
    x   2   3   x
    x   0   1   x
    O   x   x   x----->x
    """
    print("create_charuco_corners: ", board_rows, 'x', board_cols)
    print("square_length: ", square_length)
    print("marker_length: ", marker_length)
    square_length_in_meter = square_length * 0.01
    # chess board corner
    total_id_count = (board_rows - 1) * (board_cols - 1)
    charuco_corners = {}
    for i in range(total_id_count):
        x = int(i % (board_cols - 1) + 1) * square_length_in_meter
        y = int(i / (board_cols - 1) + 1) * square_length_in_meter
        charuco_corners[int(i)] = np.array([x, y])
    # for item in charuco_corners.items():
    #     print(item)
    return charuco_corners

def charuco_id_to_board_id(charuco_id, board_rows, board_cols, split_rows, split_cols):
    '''
    例如:我们生成了一个24x12的charuco_board,总id个数为23*11=253
    把它分成3x2总共6个board,每个board是8x6
    这个例子中:board_rows = 24, board_cols = 12
    split_rows = 3, split_cols = 2

    charuco_id见create_charuco_corners的描述,就是它的key
    board_id是被split后的id,在这个例子中,我们把24x12的完整board分成 3x2 共6个split_board,那么它们的id是:
    ['1-1', '1-2', '2-1', '2-2', '3-1', '3-2']
    这个id会在split_board左上角写清楚
    '''
    charuco_id = int(charuco_id)
    board_rows = int(board_rows)
    board_cols = int(board_cols)
    split_rows = int(split_rows)
    split_cols = int(split_cols)

    total_id_count = (board_rows - 1) * (board_cols - 1)
    assert(charuco_id < total_id_count)

    split_board_rows = int(board_rows / split_rows) # 8行
    split_board_cols = int(board_cols / split_cols) # 6列
    # print('split charuco board size = ', split_board_rows, 'x', split_board_cols)
    # 左下角作为原点,从左往右,从下往上
    x = int(charuco_id % (board_cols - 1) + 1)
    y = int(charuco_id / (board_cols - 1) + 1)
    if (x % split_board_cols == 0) or (y % split_board_rows == 0):
        print('invalid charuco id = ', charuco_id)
        return 'invalid'
    out_row = split_rows - int(y / split_board_rows)
    out_col = int(x / split_board_cols) + 1
    out_str = str(out_row) + '_' + str(out_col)
    return out_str

def classify_charuco_board(charuco_pts_3d_all, board_rows, board_cols, split_rows, split_cols):
    '''
    colmap的images全部检测的charuco corners三角化出来的3d点,是多个split_board的集合
    根据charuco_id即point3d_id,找到split_board_id,即这些3d点来自同一块split_board
    '''
    print('charuco board size = ', board_rows, 'x', board_cols)
    print('we split it into = ', split_rows, 'x', split_cols)
    split_board_rows = int(board_rows / split_rows) # 8行
    split_board_cols = int(board_cols / split_cols) # 6列
    print('split charuco board size = ', split_board_rows, 'x', split_board_cols)
    # key是该charuco_id所属的split_charuco_board的str名称
    # value是该split_charuco_board检测到的所有3d点及其charuco_id
    charuco_pts_3d_dict = {}
    for i in range(charuco_pts_3d_all.shape[0]):
        charuco_id = charuco_pts_3d_all[i, 3]
        out_str = charuco_id_to_board_id(charuco_id, board_rows, board_cols, split_rows, split_cols)
        # print(out_str)
        charuco_pts_3d_dict[out_str] = []
    for i in range(charuco_pts_3d_all.shape[0]):
        charuco_id = charuco_pts_3d_all[i, 3]
        out_str = charuco_id_to_board_id(charuco_id, board_rows, board_cols, split_rows, split_cols)
        charuco_pts_3d_dict[out_str].append(charuco_pts_3d_all[i, :])
    return charuco_pts_3d_dict

def charuco_pt_detection(image_path, board_parameters_yaml, min_id_pixel_distance = 1, min_ids = 1):
    # 读标定板参数
    board_rows, board_cols, square_length, marker_length = load_board_parameters(board_parameters_yaml)
    # 构造词典
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    # 构造board, 
    # 注意,入参是:squaresX, squaresY, squareLength, markerLength, dictionary
    # 所以cols在前,rows在后,trajectory_from_aruco文件夹里面用反了,那边不动,这里我纠正过来
    board = aruco.CharucoBoard_create(board_cols, board_rows, square_length, marker_length, aruco_dict)
    aruco_params = aruco.DetectorParameters_create()

    img = cv2.imread(image_path)
    if img is None:
        return None, None
    # 行, 列, 通道
    hh, ww, cc = img.shape
    hh_show = int(900)
    ww_show = int(float(ww) * float(hh_show) / float(hh))
    # 检测maker角点
    corners, ids, rejected_points = aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)
    # refine检测maker角点
    aruco.refineDetectedMarkers(img, board, corners, ids, rejected_points)
    # 画maker四个角点和id
    img_with_markers = aruco.drawDetectedMarkers(img, corners, ids, (0, 0, 255))
    if ids is not None and len(ids) >= min_ids:
        # 插值charuco角点
        charuco_retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, img, board)
        if charuco_retval != 0:
            img_with_charuco = aruco.drawDetectedCornersCharuco(img_with_markers, charuco_corners, charuco_ids, (0, 255, 0))
            img_with_charuco_resize = cv2.resize( img_with_markers, (ww_show, hh_show) )
            #visual_2
            #cv2.imshow("draw marker and charuco", img_with_charuco_resize)
            #cv2.waitKey(1000)
            # new_image_path = image_path[:-4] + '_charuco_marker_id.png'
            # cv2.imwrite(new_image_path, img_with_charuco)
            return charuco_corners, charuco_ids
        else:
            print('Charuco corners detection failed! \n please check the config of charuco board! rows and cols should be complete!')
            return None, None
        
    else:
        #print('Marker detection failed!')
        return None, None

def run_charuco_detection(colmap_image_folder, write_txt_folder, board_path_name, selected_folder = '', min_id_pixel_distance = 1, min_ids = 1): 
    selected_image_folder = colmap_image_folder + '/' + selected_folder
    image_names = os.listdir(selected_image_folder)
    image_names.sort()

    image_valid = []
    charuco_valid = []
    num_charuco_image = 0
    for image_name in image_names:
        # 只处理图像
        if image_name.endswith('.png') or image_name.endswith('.jpg') or \
            image_name.endswith('.PNG') or image_name.endswith('.JPG'):
            image_relative_path_name = os.path.join(selected_folder, image_name)
            image_path_name = os.path.join(colmap_image_folder, image_relative_path_name)
            
           # print('charuco_pt_detection from', image_path_name)
            charuco_corners, charuco_ids = charuco_pt_detection( image_path_name, board_path_name, min_id_pixel_distance, min_ids )
            if charuco_corners is None or charuco_ids is None:
                continue
            print('detect ', charuco_corners.shape, 'charuco corners')
            num_charuco_image = num_charuco_image + 1
            txt_name = image_name + '.txt'
            txt_path_name = os.path.join(write_txt_folder, txt_name)
           # write_charucos(txt_path_name, charuco_corners, charuco_ids)
            write_charucos_as_colmap_feature(txt_path_name, charuco_corners, charuco_ids)
            image_valid.append(image_relative_path_name)
            charuco_valid.append(charuco_ids)
    
    write_charucos_match(write_txt_folder + '_match.txt', image_valid, charuco_valid)
    return num_charuco_image

def get_names_in_imageList(images_list_file):
    lines = []
    with open(images_list_file) as fp:
        lines = fp.readlines()

    names = []
    for line in lines:
        line = line.strip()
        _, name = os.path.split(line)

        if len(name)>0:
            names.append(name)

    return names

def run_charuco_detection_with_imageList(colmap_image_folder, write_txt_folder, board_path_name, images_list_file, selected_folder='',
                          min_id_pixel_distance=1, min_ids=1):
    image_names = get_names_in_imageList(images_list_file)
    image_names.sort()

    image_valid = []
    charuco_valid = []
    num_charuco_image = 0
    for image_name in image_names:
        # 只处理图像
        if image_name.endswith('.png') or image_name.endswith('.jpg') or \
                image_name.endswith('.PNG') or image_name.endswith('.JPG'):
            image_relative_path_name = os.path.join(selected_folder, image_name)
            image_path_name = os.path.join(colmap_image_folder, image_relative_path_name)

            # print('charuco_pt_detection from', image_path_name)
            charuco_corners, charuco_ids = charuco_pt_detection(image_path_name, board_path_name, min_id_pixel_distance,
                                                                min_ids)
            if charuco_corners is None or charuco_ids is None:
                continue
            print('detect ', charuco_corners.shape, 'charuco corners')
            num_charuco_image = num_charuco_image + 1
            txt_name = image_name + '.txt'
            txt_path_name = os.path.join(write_txt_folder, txt_name)
            # write_charucos(txt_path_name, charuco_corners, charuco_ids)
            write_charucos_as_colmap_feature(txt_path_name, charuco_corners, charuco_ids)
            image_valid.append(image_relative_path_name)
            charuco_valid.append(charuco_ids)

    write_charucos_match(write_txt_folder + '_match.txt', image_valid, charuco_valid)
    return num_charuco_image

def get_charuco_ids_from_model(database_path, model_path, ext):
    cameras, images, points3D = read_model(model_path, ext)
    charuco_3ds = np.zeros((len(points3D), 4) , dtype = float)

    db = COLMAPDatabase.connect(database_path)
    desps = db.execute("SELECT * FROM descriptors;")
    # image_ids, kpts, dims, descriptors = next(desps)

    for i, idx in enumerate(points3D): 
        pt = points3D[idx]
        xyz = pt.xyz
        image_id1 = pt.image_ids[0]
        p2d_id1 = pt.point2D_idxs[0]
        desps = db.execute("SELECT * FROM descriptors;")
        for desp_image in desps:
            if desp_image[0] == image_id1:
                desp = desp_image[3]
                dim = desp_image[2]
                charuco_3ds[i, : ] = np.array([xyz[0], xyz[1], xyz[2], desp[p2d_id1 * dim]])
                print(idx, ' ', i, charuco_3ds[i, : ])
    return charuco_3ds
#    print(charuco_3ds)
        

def main():
    '''
    +── images //固定的名称
    │   +── folder1 //可变的名称
        │   +── image1.jpg
        │   +── image2.jpg
        │   +── ...
    │   +── folder2 //可变的名称
        │   +── image1.jpg
        │   +── image2.jpg
        │   +── ...
    +── sparse //固定的名称
    │   +── old_model //可变的名称
    │   │   +── cameras.bin
    │   │   +── images.bin
    │   │   +── points3D.bin
    │   +── ...
    │   +── new_model //可变的名称
    │   │   +── cameras.bin
    │   │   +── images.bin
    │   │   +── points3D.bin
    │   +── ...
    +── database.db //可变的名称
    '''
    if len(sys.argv) < 4:
        print('detect charuco corners for all images in a folder.')
        print('python charuco_detection.py [colmap_image folder] [save_txt folder] [board_parameters yaml] [(opt)min_id_pixel_distance] [(opt)min_ids]')
        return
    # colmap工程的根目录
    colmap_image_folder = sys.argv[1]
    write_txt_folder = sys.argv[2]
    board_path_name = sys.argv[3]
    min_id_pixel_distance = 1
    min_ids = 1
    if len(sys.argv) >= 5:
        min_id_pixel_distance = int(sys.argv[4])
    if len(sys.argv) >= 6:
        min_ids = int(sys.argv[5])
 
    run_charuco_detection(colmap_image_folder, write_txt_folder, board_path_name, '', min_id_pixel_distance, min_ids)
    
    print('All done!')

if __name__ == '__main__':
    main()
