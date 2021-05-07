# coding: utf-8
import os
import sys
import argparse
import numpy as np
import random

def random_positions():
    total_count = 20
    sample_range = [-5.0, 5.0]
    positions_2d = np.zeros((total_count, 2))
    for i in range(total_count):
        x = random.uniform(sample_range[0], sample_range[1])
        y = random.uniform(sample_range[0], sample_range[1])
        positions_2d[i, 0] = x
        positions_2d[i, 1] = y
    # 保证p0是原点, p1在x轴正方向上
    positions_2d[0, 0] = 0.0
    positions_2d[0, 1] = 0.0
    positions_2d[1, 0] = np.abs(positions_2d[1, 0]) + 1.0
    positions_2d[1, 1] = 0.0
    # for i in range(total_count):
    #     print('id = ', i, positions_2d[i,:])
    return positions_2d

def is_pt_upper_vec(p0, p1, p2):
    '''
    计算点p2, 是在向量vec01的逆时针方向True, 还是顺时针方向False
    '''
    vec01 = p1 - p0
    vec02 = p2 - p0
    xp = vec01[0] * vec02[1] - vec01[1] * vec02[0]
    if xp >= 0.0:
        return True
    else:
        return False

def to_line_str(positions_2d, id_begin, id_end, is_up):
    delta = positions_2d[id_end, : ] - positions_2d[id_begin, : ]
    noise = random.gauss(0.0, 0.03)
    # print(noise)
    length = np.linalg.norm(delta) + noise
    return str(id_begin) + ' ' + str(id_end) + ' ' + str(length) + ' ' + str(is_up) + '\n'

def read_dis2d(dis2d_path):
    infile = open(dis2d_path, 'r')
    dis_list = []
    for line in infile.readlines(): # 依次读取每行  
        line = line.strip() # 去掉每行头尾空白
        if line == '# loop edges':
            print('break')
            break
        if line[0] == '#':
            continue
        elements = line.split(' ')
        is_up = True
        if int(elements[3]) < 0:
            is_up = False
        # 保证distance的两个id, 前者小于后者
        if int(elements[0]) < int(elements[1]):
            dis_list.append([int(elements[0]), int(elements[1]), float(elements[2]), is_up])
        else:
            dis_list.append([int(elements[1]), int(elements[0]), float(elements[2]), is_up])
    infile.close()
    return dis_list

def write_random_positions(dis2d_path, positions_2d):
    # id_begin id_end distance is_up_end
    outfile = open(dis2d_path, 'w')
    outfile.write('# id_begin id_end distance is_up_end \n')
    outfile.write(to_line_str(positions_2d, 0, 1, 1))
    for i in range(2, positions_2d.shape[0]):
        is_up = is_pt_upper_vec(positions_2d[i-2], positions_2d[i-1], positions_2d[i])
        is_up_int = 1
        if not is_up:
            is_up_int = -1
        outfile.write(to_line_str(positions_2d, i-2, i, is_up_int))
        outfile.write(to_line_str(positions_2d, i-1, i, is_up_int))
    outfile.write('# loop edges \n')

    outfile.close()
    return

def cal_p2_by_p0_p1(d01, d02, d12, is_up):
    '''
    用余弦定理, 根据三个边长, 求夹角; 再根据夹角, 求坐标
    以p0为原点, p0->p1为x轴, 构建平面直角坐标系
    求p2, 需被告知p2的y是正是负
    余弦定理:
    p2 = [x2, y2] # 待求
    cos(theta201) = (d01^2 + d02^2 - d12^2) / (2 * d01 *d02)
    x2 = d02 * cos(theta201)
    y2 = d02 * sin(theta201)
    '''
    assert d01 + d02 > d12, 'd01 + d02 <= d12'
    assert d01 + d12 > d02, 'd01 + d12 <= d02'
    assert d02 + d12 > d01, 'd02 + d12 <= d01'
    # p0 = [0.0, 0.0]
    # p1 = [d01, 0.0]
    cos_theta201 = (d01*d01 + d02*d02 - d12*d12) / (2 * d01 * d02)
    theta201 = np.arccos(cos_theta201)
    sin_theta201 = np.sin(theta201)
    x2 = d02 * cos_theta201
    y2 = d02 * sin_theta201
    if is_up:
        return np.array([x2, y2])
    return np.array([x2, -y2])

def cal_src_to_x_axis(p_src0, p_src1):
    '''
    counterclockwise
    计算角度本身是逆时针, 因为我们用右手坐标系
    '''
    x_axis = np.array([1.0, 0.0])
    vec_src = p_src1 - p_src0
    vec_src = vec_src / np.linalg.norm(vec_src)
    dot = x_axis[0] * vec_src[0] + x_axis[1] * vec_src[1]
    det = x_axis[1] * vec_src[0] - vec_src[1] * x_axis[0]
    theta = np.arctan2(det, dot)
    return [theta, p_src0]

def to_transform(deg_trans):
    '''
    clockwise
    transform a vector to x-axis
    把一个vector旋转到跟x-axis平行, 需要顺时针
    角度本身是逆时针, 需要顺时针把它纠正回来
    '''
    theta = deg_trans[0]
    transfrom_mat = np.eye(3)
    transfrom_mat[0:2, 0:2] = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    transfrom_mat[0, 2] = deg_trans[1][0]
    transfrom_mat[1, 2] = deg_trans[1][1]
    return transfrom_mat

def transform_point(transfrom_mat, point):
    point_hom = np.array([point[0], point[1], 1.0])
    pt_new = np.matmul(transfrom_mat, point_hom)
    return np.array([pt_new[0], pt_new[1]])

def cal_pos_by_ranges(dis2d):
    for item in dis2d:
        print(item)
    # 确定原点和x轴
    positions_dict = {}
    positions_dict[0] = np.array([0.0, 0.0])
    positions_dict[1] = np.array([dis2d[0][2], 0.0])
    # 迭代式的计算所有点的坐标
    
    for i in range(2, len(dis2d), 2):
        d01 = dis2d[i-2][2]
        p0_id = dis2d[i-2][0]
        p1_id = dis2d[i-2][1]

        d02 = dis2d[i-1][2]
        error_str = str(dis2d[i-1][0]) + ' ' + str(p0_id)
        assert dis2d[i-1][0] == p0_id, error_str
        p2_id = dis2d[i-1][1]

        d12 = dis2d[i][2]
        assert dis2d[i][0] == p1_id
        assert dis2d[i][1] == p2_id

        assert p0_id in positions_dict
        assert p1_id in positions_dict

        if p2_id in positions_dict:
            print('Warning, ', p2_id, ' has been calculated!')
            print("p2_id = ", p2_id, " value = ", positions_dict[p2_id])
            continue
        p2 = cal_p2_by_p0_p1(d01, d02, d12, dis2d[i][3])
        transform_mat = to_transform( cal_src_to_x_axis(positions_dict[p0_id], positions_dict[p1_id]) )
        p2_new = transform_point(transform_mat, p2)
        positions_dict[p2_id] = p2_new
    return positions_dict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dis2d_path', required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    positions_2d = random_positions()
    write_random_positions(args.dis2d_path, positions_2d)

    dis2d_list = read_dis2d(args.dis2d_path)
    positions_dict = cal_pos_by_ranges(dis2d_list)

    delta_list = []
    for key, value in positions_dict.items():
        delta = np.linalg.norm(value - positions_2d[key, : ])
        delta_list.append(delta)
        print(key, value, ' vs ', positions_2d[key, : ], ', delta = ', delta)
    delta_list = np.array(delta_list)
    print('mean = ', np.mean(delta_list))
if __name__ == "__main__":
    main()