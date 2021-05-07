# coding: utf-8
import math
import numpy as np
import argparse
import sys
import copy
import cv2
sys.path.append("../")

from trajectory_tools.colmap_to_tum_evo import read_geo

def get_point_callback(event, x, y, flags, param):
    """
    cv2鼠标回调函数
    """
    global points
    if event == cv2.EVENT_MBUTTONDOWN:
        points.append([x, y])

def draw_points(im, p):
    """
    绘制函数
    """
    img_new = copy.deepcopy(im)
    for i in range(len(p)):
        cv2.circle(img_new, (p[i][0], p[i][1]), 3, (0, 0, 255), -1) # 画点
        if i > 0:
            cv2.line(img_new, (p[i-1][0], p[i-1][1]), (p[i][0], p[i][1]), (255, 255, 0), 2) # 画线
    return img_new

def draw_arrows(img, p):
    img_new = copy.deepcopy(img)
    for i in range(0, len(p)):
        cv2.circle(img_new, (p[i][0], p[i][1]), 3, (0, 255, 255), -1) # 画点
        if i > 0 and (i+1) % 2 == 0:
            cv2.arrowedLine(img_new, (p[i-1][0], p[i-1][1]), (p[i][0], p[i][1]), (255, 255, 0), 5)
    return img_new

def read_image_config(image_config_path, image_type):
    in_file = open(image_config_path,'r')
    resolution = 0.0
    max_row = 0.0
    max_col = 0.0
    for line in in_file.readlines(): # 依次读取每行  
        line = line.strip() # 去掉每行头尾空白
        if line[0] == '#':
            continue
        elements = line.split(':')
        if elements[0] == 'resolution(m)':
            resolution = float(elements[1])
        else:
            if image_type == 'XY':
                if elements[0] == 'max_X_ROW':
                    max_row = int(elements[1])
                elif elements[0] == 'max_Y_COL':
                    max_col = int(elements[1])
                else:
                    print('Error image config:', elements[0])
                    exit(0)
            elif image_type == 'ZX':
                if elements[0] == 'max_Z_ROW':
                    max_row = int(elements[1])
                elif elements[0] == 'max_X_COL':
                    max_col = int(elements[1])
                else:
                    print('Error image config:', elements[0])
                    exit(0)
            elif image_type == 'ZY':
                if elements[0] == 'max_Z_ROW':
                    max_row = int(elements[1])
                elif elements[0] == 'max_Y_COL':
                    max_col = int(elements[1])
                else:
                    print('Error image config:', elements[0])
                    exit(0)
            else:
                print('Error image type:', image_type)
                exit(0)
    return [resolution, max_row, max_col]

def image_matrix_to_world(points, image_config):
    resolution = image_config[0]
    max_row = image_config[1]
    max_col = image_config[2]
    points = np.array(points)
    image_rowcol = np.zeros(points.shape)
    image_rowcol[:,0] = points[:,1]
    image_rowcol[:,1] = points[:,0]
    world_rowcol = resolution * ([max_row, max_col] -  image_rowcol)
    return world_rowcol

def anticlockwise_degree_u2v(u1, u2, v1, v2):
    '''
    https://zh.wikipedia.org/wiki/%E5%8F%89%E7%A7%AF
    u X v = (u2v3 - u3v2)i + (u3v1 - u1v3)j + (u1v2 - u2v1)k
    把u和v的z都设为0,得到: u X v = u1v2 - u2v1
    在转向角度为180°以内的条件下,
    若 u X v > 0,说明是绕逆时针,a转向b
    若 u X v < 0,说明是绕顺时针,a转向b
    若 u X v = 0,说明a和b平行,至于是0还是180,看arccos
    u dot v = |u||v|cos(theta)
    theta = arccos(u dot v)
    theta->[0, 180], 加上符号后, theta->[0, 360]
    返回单位向量u和角度约束theta
    如果某个向量在xy平面跟u计算角度([0,360])在theta范围内,说明符合我们的约束
    '''
    u = np.array([u1, u2])
    u = u / np.linalg.norm(u)
    v = np.array([v1, v2])
    v = v / np.linalg.norm(v)
    theta = math.acos(np.dot(u, v)) * 57.29577951308232
    if u1 * v2 - u2 * v1 < 0:
        theta = 360.0 - theta
    return [u[0], u[1], theta]

def write_position_constraint(out_path, image_type, world_rowcol):
    '''
    标注位置约束的时候,如果最后一个点跟第一个点的像素非常接近,则退出,因此我们不需要保存最后一个点
    因为在构建polygon的时候,默认最后一个点连接第一个点
    '''
    outfile = open(out_path, 'w')
    line_str = '#<world coordinate>rowcol:' + image_type + '\n'
    outfile.write(line_str)
    outfile.write('#<polygon points list>\n')
    # 不需要最后一个点
    for i in range(world_rowcol.shape[0] - 1):
        line_str = str(world_rowcol[i, 0]) + ' ' + str(world_rowcol[i, 1]) + '\n'
        outfile.write(line_str)
    outfile.close()
    return

def read_position_constraint(in_path):
    infile = open(in_path, 'r')
    world_coordinate = None
    polygon_pt_list = []
    for line in infile.readlines(): # 依次读取每行  
        line = line.strip() # 去掉每行头尾空白
        strs = line.split(':')
        if strs[0] == '#<world coordinate>rowcol':
            world_coordinate = strs[1]
            continue
        if line[0] == '#':
            continue
        strs = line.split(' ')
        polygon_pt_list.append([float(strs[0]), float(strs[1])])
    return world_coordinate, polygon_pt_list

def write_degree_constraint(out_path, image_type, world_rowcol):
    world_rowcol = np.array(world_rowcol)
    u = world_rowcol[1,:] - world_rowcol[0,:]
    v = world_rowcol[3,:] - world_rowcol[2,:]
    deg_constraint = anticlockwise_degree_u2v(u[0], u[1], v[0], v[1])
    outfile = open(out_path, 'w')
    line_str = '#<world coordinate>rowcol:' + image_type + '\n'
    outfile.write(line_str)
    outfile.write('#<config>:unit_vector theta([0.0, 360.0])\n')
    outfile.write('#<current_degree>:anti-clockwise from unit_vector to orientation\n')
    outfile.write('#<degree_constraint>:current_degree <= theta\n')
    deg_constraint_str = str(deg_constraint[0]) + ' ' + str(deg_constraint[1])  + ' ' + str(deg_constraint[2])
    outfile.write(deg_constraint_str)
    outfile.close()
    return

def read_degree_constraint(in_path):
    infile = open(in_path, 'r')
    world_coordinate = None
    degree_constraint = []
    for line in infile.readlines(): # 依次读取每行  
        line = line.strip() # 去掉每行头尾空白
        strs = line.split(':')
        if strs[0] == '#<world coordinate>rowcol':
            world_coordinate = strs[1]
            continue
        if line[0] == '#':
            continue
        strs = line.split(' ')
        degree_constraint = [float(strs[0]), float(strs[1]), float(strs[2])]
    return world_coordinate, degree_constraint

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path' , required=True)
    parser.add_argument('--label_type' , type=str, default='polygon')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.label_type != 'polygon' and args.label_type != 'vector':
        print('label_type Only support <polygon>, <vector>')
        exit(0)
    image_type = args.image_path[-6:-4]
    if image_type != 'XY' and image_type != 'ZX' and image_type != 'ZY':
        print('image should be endwith <_XY.png>, <_ZX.png>, <_ZY.png>')
        exit(0)
    
    img = cv2.imread(args.image_path)
    image_config_path = args.image_path[0:-4] + '.txt'
    image_config = read_image_config(image_config_path, image_type)

    strs = args.image_path.split('/')
    imshow_name = strs[-1] + '_' + args.label_type
    image_drawn_path = args.image_path[0:-4] + '_' + args.label_type + '.png'
    txt_drawn_path = args.image_path[0:-4] + '_' + args.label_type + '.txt'

    global points
    points = []

    cv2.namedWindow(imshow_name)
    cv2.setMouseCallback(imshow_name, get_point_callback)

    img_new = None

    if args.label_type == 'polygon':
        while (True):
            img_new = draw_points(img, points)
            cv2.imshow(imshow_name, img_new)
            if len(points) > 1:
                if ( abs(points[0][0]-points[-1][0]) + abs(points[0][1]-points[-1][1]) )<10:
                    points.pop(-1)
                    break
            key = cv2.waitKey(10)
            if key == 27:
                'Esc'
                break
            elif key == 114:
                'r: Clear all'
                print('Clear All!')
                points = []
            elif key == 32:
                '空格：撤回一步操作'
                if len(points) > 0 :
                    print("Pop one point ! ")
                    points.pop(-1)
                else :
                    print("No points!")
        if len(points) < 3:
            print('Error->polygon, less than 3 points')
            exit(0)
    elif args.label_type == 'vector':
        while (True):
            img_new = draw_arrows(img, points)
            cv2.imshow(imshow_name, img_new)
            if len(points) == 4:
                # cv2.waitKey(1000)
                break
            key = cv2.waitKey(10)
            if key == 27:
                'Esc'
                break
            elif key == 114:
                'r: Clear all'
                print('Clear All!')
                points = []
            elif key == 32:
                '空格：撤回一步操作'
                if len(points) > 0 :
                    print("Pop one point ! ")
                    points.pop(-1)
                else :
                    print("No points!")
        if len(points) < 4:
            print('Error->vector, less than 4 points')
            exit(0)
    else:
        print('label_type Only support <polygon>, <vector>')
        exit(0)

    world_rowcol = image_matrix_to_world(points, image_config)
    if args.label_type == 'polygon':
        write_position_constraint(txt_drawn_path, image_type, world_rowcol)
    elif args.label_type == 'vector':
        write_degree_constraint(txt_drawn_path, image_type, world_rowcol)
    else:
        print('label_type Only support <polygon>, <vector>')
        exit(0)
    cv2.imwrite(image_drawn_path, img_new)
    cv2.destroyAllWindows()