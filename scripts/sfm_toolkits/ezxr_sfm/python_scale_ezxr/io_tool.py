# coding: utf-8
import os
import numpy as np
import yaml

from colmap_process import colmap_read_write_model
#import read_write_model

def load_board_parameters(yaml_name):
    try:
        f = open(yaml_name)
        next(f)  # 跳过第一行
    except UnicodeDecodeError:
        f = open(yaml_name, encoding='UTF-8')
        next(f)
    content = yaml.load(f)

    board_rows = content['Board.rows']
    board_cols = content['Board.cols']
    square_length = content['Board.square_length']
    marker_length = content['Board.marker_length']

    return board_rows, board_cols, square_length, marker_length

def read_cameras_and_images(model_folder, ext):
    '''
    从colmap的model读入所有poses和camera intrinsic

    :param model_folder:到colmap的model的路径,里面应该包含cameras.bin/txt, images.bin/txt, points3D.bin/txt
    :param ext:         文件尾缀,txt是明文,其他是二进制
    :return:            cameras模型, images模型,见colmap的文档说明
    '''
    if ext == ".txt":
        cameras = read_write_model.read_cameras_text(os.path.join(model_folder, "cameras" + ext))
        images = read_write_model.read_images_text(os.path.join(model_folder, "images" + ext))
    else:
        cameras = read_write_model.read_cameras_binary(os.path.join(model_folder, "cameras" + ext))
        images = read_write_model.read_images_binary(os.path.join(model_folder, "images" + ext))
    return cameras, images

def read_undistortion_points_2d(points_2d_folder):
    image_names = os.listdir(points_2d_folder)
    image_names.sort()
    points_2d_dict = {}
    for image_name in image_names:
        if image_name.endswith('.txt'):
            image_path_name = os.path.join(points_2d_folder, image_name)
            pts_2d = np.loadtxt(image_path_name, dtype=float)
            points_2d_dict[image_name[0:-4]] = pts_2d
    return points_2d_dict

def write_charucos(txt_path_name, charuco_corners, charuco_ids):
    '''
    输出格式:u v id; u v是像素坐标,id是charuco_ids
    '''
    assert (charuco_corners.shape[0] == charuco_ids.shape[0])
    charucos = np.zeros( (charuco_corners.shape[0], 3) , dtype = float)
    for i in range(charuco_corners.shape[0]):
        charucos[i, : ] = np.array([charuco_corners[i, 0, 0], charuco_corners[i, 0, 1], charuco_ids[i, 0]])
    np.savetxt(txt_path_name, charucos, fmt='%f')
    return

def write_charucos_as_colmap_feature(txt_path_name, charuco_corners, charuco_ids):
    '''
    输出格式: 点数 128
    u v id 1 1 1 ...; u v是像素坐标,id是charuco_ids，后续127个1是补足colmap128维描述子
    '''
    assert (charuco_corners.shape[0] == charuco_ids.shape[0])
    outfile = open(txt_path_name, 'w+')

    outfile.write(str(charuco_corners.shape[0]) + ' 128\n')

    charucos = np.zeros( (charuco_corners.shape[0], 130) , dtype = float)
    for i in range(charuco_corners.shape[0]):
        charucos[i, 0] = charuco_corners[i, 0, 0]
        charucos[i, 1] = charuco_corners[i, 0, 1]
        charucos[i, 2:] = charuco_ids[i, 0]
    
    for fea in charucos:
        for value in fea:
            outfile.write(str(value) + ' ')  #\r\n为换行符
        outfile.write('\n')

    outfile.close()

def write_charucos_match(txt_path_name, images, charuco_ids):
    outfile = open(txt_path_name, 'w+')
    for i, name1 in enumerate(images):
        for j, name2 in enumerate(images):
            if j <= i:
                continue

            ids1 = charuco_ids[i]
            ids2 = charuco_ids[j]

            keypoint_matches = []
            for a, fea1 in enumerate(ids1):
                for b, fea2 in enumerate(ids2):
                    if fea1 == fea2:
                        keypoint_matches.append([a, b])

            if len(keypoint_matches) > 5:
                outfile.write(name1 + ' ' + name2 + '\n')
                for (match1, match2) in keypoint_matches:
                    outfile.write(str(match1) + ' ' + str(match2) + '\n')
                outfile.write('\n')