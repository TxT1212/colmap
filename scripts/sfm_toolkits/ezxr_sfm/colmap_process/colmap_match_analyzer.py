# coding: utf-8
import os
import sys
import math
import numpy as np
import argparse
import shutil
import networkx as nx # 图, 相关算法
import matplotlib.pyplot as plt
sys.path.append('../')
from colmap_process.colmap_db_parser import *
from colmap_process.colmap_read_write_model import *
from colmap_process.colmap_keyframe_selecter import auto_read_model
from colmap_process.colmap_export_geo import quaternion_matrix

def construct_match_graph_database(database_path, match_inlier_threshold):
    '''
    CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB)
    '''
    '''
    从database里读取match信息，即two_view_geometries
    这里的match只使用了两视图几何信息
    属于先验信息
    '''
    print('construct_match_graph_database...')
    match_graph_db = nx.Graph()
    db = COLMAPDatabase.connect(database_path)
    cursor = db.cursor()
    cursor.execute("select * from two_view_geometries")
    two_view_geometries = cursor.fetchall()
    for match in two_view_geometries:
        if int(match[1]) >= match_inlier_threshold:
            image_id1, image_id2 = pair_id_to_image_ids(match[0])
            image_id1 = str(int(image_id1))
            image_id2 = str(int(image_id2))
            match_graph_db.add_node(image_id1)
            match_graph_db.add_node(image_id2)
            match_graph_db.add_edge(image_id1, image_id2)
    db.close()
    print('node number = ', len(match_graph_db.nodes()))
    print('edge number = ', len(match_graph_db.edges()))
    return match_graph_db

def construct_match_graph_model(model_path, match_graph_db, match_inlier_threshold):
    '''
    从model里读取match信息，两幅图像共同观测的3d点大于等于match_inlier_threshold认为是match
    这里的match使用了3d信息
    属于后验信息
    '''
    print('construct_match_graph_model...')
    if model_path[-1] != '/':
        model_path = model_path + '/'
    _, images, point3ds = auto_read_model(model_path)
    hist_path = model_path + 'track_length_hist.png'
    print('--->track_length_hist...')
    track_length_list = []
    for _, pt in point3ds.items():
        track_num = len(pt.image_ids)
        if track_num > 10:
            track_num = 10
        track_length_list.append(track_num)
    plt.hist(track_length_list, bins=[0,2,4,6,8,10])
    plt.savefig(hist_path, dpi=500, bbox_inches='tight')
    plt.close()
    match_graph_model = nx.Graph()
    for key, _ in images.items():
        match_graph_model.add_node(str(key))
    assert len(match_graph_model.nodes()) <= len(match_graph_db.nodes()), 'Error! database and model do not match'
    print('node number = ', len(match_graph_model.nodes()))
    idx = 0
    delta = int(len(match_graph_db.edges()) * 0.02) * 10
    for aa, bb in match_graph_db.edges():
        if idx % delta == 0:
            print('processed/total: ', idx, '/', len(match_graph_db.edges()))
        aa = int(aa)
        bb = int(bb)
        if aa not in images or bb not in images:
            idx += 1
            continue
        aa_pt3ds = set(images[aa].point3D_ids)
        bb_pt3ds = set(images[bb].point3D_ids)
        aa_bb_inter = aa_pt3ds.intersection(bb_pt3ds)
        if len(aa_bb_inter) >= match_inlier_threshold:
            match_graph_model.add_edge(str(aa), str(bb))
        idx += 1
    
    print('edge number = ', len(match_graph_model.edges()))
    return match_graph_model, images

def broken_match_analyzer(match_graph_db, match_graph_model):
    '''
    match_graph_model是match_graph_db的子集
    无论node还是edge，前者都是后者的子集
    '''
    print('broken_match_analyzer...')
    broken_match_dict = {}
    for node in match_graph_model.nodes():
        degree_model = match_graph_model.degree[node]
        degree_db = match_graph_db.degree[node]
        broken_match_dict[node] = degree_db - degree_model
    return broken_match_dict

def write_geos_with_broken_match(model_path, broken_match_dict, images):
    print('write_geos_with_broken_match...')
    print('--->broken_match_hist...')
    if model_path[-1] != '/':
        model_path = model_path + '/'
    hist_path = model_path + 'broken_match_hist.png'
    geos_path = model_path + 'geos_with_broken_match.txt'
    broken_match_list = []
    for key, value in broken_match_dict.items():
        broken_match_list.append(value)
    plt.hist(broken_match_list)
    plt.savefig(hist_path, dpi=500, bbox_inches='tight')
    plt.close()
    print('--->write_geos...')
    fid = open(geos_path, 'w')
    for key, img in images.items():
        rmat = quaternion_matrix(img.qvec).T
        tvec = img.tvec
        tnew = -rmat @ tvec
        image_header = [img.name, *tnew, broken_match_dict[str(key)]]
        first_line = " ".join(map(str, image_header))
        fid.write(first_line + "\n")
    fid.close()
    return

def write_track_length_hist(model_path):
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_path', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--match_inlier_threshold', type=int, default=15)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    match_graph_db = construct_match_graph_database(args.database_path, args.match_inlier_threshold)
    match_graph_model, images = construct_match_graph_model(args.model_path, match_graph_db, args.match_inlier_threshold)
    broken_match_dict = broken_match_analyzer(match_graph_db, match_graph_model)
    write_geos_with_broken_match(args.model_path, broken_match_dict, images)
    return

if __name__ == '__main__':
    main()