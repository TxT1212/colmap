# coding: utf-8
import os
import sys
import argparse
import shutil
import copy
import numpy as np
import networkx as nx # 图, 相关算法
import matplotlib.pyplot as plt
sys.path.append('../')

from colmap_process.colmap_model_analyzer import get_folders_in_folder
from colmap_process.colmap_map_merger_plus import run_database_merger
from colmap_process.colmap_seq_sfm import run_model_sim3_merger
from colmap_process.colmap_model_modify import update_loc_model_id_refer_to_locmap_database

from colmap_process_loop_folders.colmap_models_analyzer import read_model_report_status, read_model_reports
from colmap_process_loop_folders.tree_topology import getSceneTopology, getTopologyCheckTasks,\
    getTopologyCheckTasksSpecified, getOnlyNamesFromNodesInfo, filterTopologyCheckTasks, cleanSceneTopology,\
    writeSceneTolopogy

from python_scale_ezxr.transformations import quaternion_from_matrix, quaternion_matrix

def decompose_sim3_transform(sim3_transform):
    '''
    把sim3矩阵分解成scale, rotation和translation
    sim3矩阵的3*3部分是scale和rotation相乘后的结果
    scale = norm(scale_rot_row0) = norm(scale_rot_row1) = norm(scale_rot_row2)
    '''
    scale_rot_row0 = sim3_transform[0, 0:3]
    # scale_rot_row1 = sim3_transform[1, 0:3]
    # scale_rot_row2 = sim3_transform[2, 0:3]
    scale = np.linalg.norm(scale_rot_row0)
    print('scale_rot_row0 = ', scale_rot_row0)
    print('scale = ', scale)
    rmat = sim3_transform[0:3, 0:3] / scale
    print('rmat = ', rmat)
    quat = quaternion_from_matrix(rmat)
    tvec = sim3_transform[0:3, 3]
    return scale, quat, tvec

def compose_sim3_transform(scale, quat, tvec):
    rmat = quaternion_matrix(quat)
    rmat = scale * rmat[0:3, 0:3]
    sim3_transform = np.identity(4)
    sim3_transform[0:3, 0:3] = rmat
    sim3_transform[0:3, 3] = np.array(tvec)
    return sim3_transform

def read_sim3_txt(sim3_txt_path):
    with open(sim3_txt_path) as f:
        content = f.readlines()
        if len(content) == 0:
            return None, None
        line_str = content[2].strip()
        line_strs = line_str.split(':')
        inlier_number = int(line_strs[1])
        sim3_transform = np.identity(4)
        for i in range(4,7):
            line_str = content[i].strip()
            line_strs = line_str.split(' ')
            sim3_transform[i - 4, 0] = float(line_strs[0])
            sim3_transform[i - 4, 1] = float(line_strs[1])
            sim3_transform[i - 4, 2] = float(line_strs[2])
            sim3_transform[i - 4, 3] = float(line_strs[3])
        return inlier_number, sim3_transform

def construct_nx_graph_sim3(full_route_list, full_routepair_list, sub_model_sim3_dict, suffix_str=''):
    current_graph = nx.Graph()

    # 失败的route也不要了，因为我们要做最大生成树
    for route_name in full_route_list:
        if sub_model_sim3_dict[route_name + suffix_str][0] == 'success':
            current_graph.add_node(route_name, success=1, color='g')
    
    # 添加成功的边
    for routepair_name in full_routepair_list:
        aa = routepair_name[0]
        bb = routepair_name[1]
        aa_bb = aa + '_' + bb + suffix_str
        if sub_model_sim3_dict[aa_bb][0] == 'success':
            current_graph.add_edge(aa, bb, weight=sub_model_sim3_dict[aa_bb][1], success=1, color='g')
    
    # 删除孤立的节点，就算节点成功了，但是它可能没有任何一条边成功
    isolated_nodes = []
    nodes = current_graph.nodes()
    for node in nodes:
        node_degree = current_graph.degree(node)
        if node_degree == 0:
            isolated_nodes.append(node)
    for node in isolated_nodes:
        current_graph.remove_node(node)
    
    # sim3_pose_graph那边的node_id是uint, 这里要把node的key从str映射到uint
    nodes = current_graph.nodes()
    idx = 0
    node_id_dict = {}
    for node in nodes:
        node_id_dict[node] = idx
        idx += 1
    # node之间的constraint，提前计算好id对应关系
    # 添加成功的边
    node_constraint_list = []
    for routepair_name in full_routepair_list:
        aa = routepair_name[0]
        bb = routepair_name[1]
        aa_bb = aa + '_' + bb + suffix_str
        if sub_model_sim3_dict[aa_bb][0] == 'success':
            node_constraint_list.append( [node_id_dict[aa], node_id_dict[bb], aa, bb, sub_model_sim3_dict[aa_bb][2]] )
    return current_graph, node_id_dict, node_constraint_list

def get_nx_tree_with_root(current_graph):
    max_tree = nx.algorithms.tree.mst.maximum_spanning_tree(current_graph, weight='weight', algorithm='prim')
    is_a_tree = nx.algorithms.tree.recognition.is_tree(max_tree)
    assert is_a_tree, 'Error! there are isolated_nodes in current_graph!'
    # 找到最适合做root的节点
    nodes_subgraph = nx.algorithms.centrality.subgraph_centrality(max_tree)
    max_score = -1.0
    max_node = None
    for key, value in nodes_subgraph.items():
        if value > max_score:
            max_score = value
            max_node = key
    return max_tree, max_node

def visualize_graph_and_tree(model_path, current_graph, max_tree, suffix_str2=''):
    sim3_construction_model_path = os.path.join(model_path, 'sim3_construction_model')
    if not os.path.isdir(sim3_construction_model_path):
        os.mkdir(sim3_construction_model_path)
    # 不同的layout影响绘图时nodes的分布摆放
    # pos = nx.spectral_layout(scene_graph)
    # pos = nx.spring_layout(scene_graph)
    pos = nx.kamada_kawai_layout(current_graph)
    nodes = current_graph.nodes()
    edges = current_graph.edges()
    node_colors = [current_graph.nodes[node]['color']  for node in nodes]
    edge_colors = [current_graph.edges[edge]['color']  for edge in edges]
    # 统一绘图
    nx.draw(current_graph, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors, font_size=6)
    # plt.show()
    save_name = sim3_construction_model_path + '/graph_current' + suffix_str2 + '.png'
    plt.savefig(save_name, dpi=500, bbox_inches='tight')

    plt.close()
    current_nodes = max_tree.nodes()
    current_node_colors = [max_tree.nodes[node]['color']  for node in current_nodes]
    nx.draw(max_tree, pos, with_labels=True, node_color=current_node_colors, font_size=6)
    save_name = sim3_construction_model_path + '/max_tree' + suffix_str2 + '.png'
    plt.savefig(save_name, dpi=500, bbox_inches='tight')
    return

def get_all_sim3_to_root(max_tree, max_node, sub_model_sim3_dict, suffix_str=''):
    all_path = nx.shortest_path(max_tree, target=max_node)
    nodes = max_tree.nodes()
    sim3_tree_dict = {}
    for node in nodes: # 循环所有叶子节点
        if node == max_node: # 起点是root跳过
            continue
        sim3_leaf2root = np.identity(4)
        for idx in range(1, len(all_path[node])): # 循环当前叶子节点到根节点的两两pair
            cur_node = all_path[node][idx]
            pre_node = all_path[node][idx - 1]
            a_str = pre_node + '_' + cur_node + suffix_str # 字符上，是from后者to前者，所以前者是map，后者是loc
            b_str = cur_node + '_' + pre_node + suffix_str
            sim3_transform = None
            # 不知道已经计算的sim3-pair是a2b还是b2a，都试一下
            if a_str in sub_model_sim3_dict:
                sim3_transform = sub_model_sim3_dict[a_str][2]
                inv_sim3 = np.linalg.inv(sim3_transform) # 已有的sim3是反的，需要求逆
                sim3_leaf2root = np.matmul(inv_sim3, sim3_leaf2root) # 左乘
            elif b_str in sub_model_sim3_dict:
                sim3_transform = sub_model_sim3_dict[b_str][2]
                sim3_leaf2root = np.matmul(sim3_transform, sim3_leaf2root) # 左乘
            else:
                raise Exception('Route-pair Error: %s\n' % a_str)
        # 遵守规则，tgt-src，后者to前者
        leaf2root_str = max_node + '+' + all_path[node][0] + suffix_str
        sim3_tree_dict[leaf2root_str] = sim3_leaf2root
    return sim3_tree_dict

def read_model_and_sim3_txts(model_folder, route_list=None, routepair_list=None, suffix_str=''):
    if model_folder[-1] != '/':
        model_folder = model_folder + '/'
    sub_model_folders = []
    sub_model_sim3_dict = {}

    # 如果没有传入额外信息，就遍历文件夹
    if route_list is None or routepair_list is None:
        sub_model_folders = get_folders_in_folder(model_folder)
    else: # 否则就按照给定的route，routepair的文件夹，进行遍历
        for route in route_list:
            route_str = route + suffix_str # 适配是否为_kf
            sub_model_folders.append(route_str)
        for routepair in routepair_list:
            routepair_str = routepair[0] + '_' + routepair[1] + suffix_str # 适配是否为_kf
            sub_model_folders.append(routepair_str)
    
    for sub_model_folder in sub_model_folders:
        full_model_path = model_folder + sub_model_folder
        if not os.path.isdir(full_model_path):
            print('No model folder named: ', full_model_path)
            sub_model_sim3_dict[sub_model_folder] = [None, None, None]
            continue
        report_path = full_model_path + '/model_report.txt'
        model_status = read_model_report_status(report_path)
        sim3_txt_path = full_model_path + '/sim3.txt'
        if os.path.isfile(sim3_txt_path):
            inlier_number, sim3_transform = read_sim3_txt(sim3_txt_path)
            sub_model_sim3_dict[sub_model_folder] = [model_status, inlier_number, sim3_transform]
        else:
            sub_model_sim3_dict[sub_model_folder] = [model_status, None, None]
    return sub_model_sim3_dict

def write_all_sim3_to_root(sim3_construction_path, sim3_tree_dict):
    if not os.path.isdir(sim3_construction_path):
        os.mkdir(sim3_construction_path)
    for key, value in sim3_tree_dict.items():
        pair_path = os.path.join(sim3_construction_path, key)
        if not os.path.isdir(pair_path):
            os.mkdir(pair_path)
        sim3_txt_path = os.path.join(pair_path, 'sim3.txt')
        sim3_mat3x4 = value[0:3, : ]
        # colmap那边的io是不认#注释，所以直接写3*4的矩阵即可
        np.savetxt(sim3_txt_path, sim3_mat3x4)
    return

def update_sim3s_by_pose_graph(sim3_construction_path, sim3_tree_dict_spg):
    for key, value in sim3_tree_dict_spg.items():
        pair_path = os.path.join(sim3_construction_path, key)
        sim3_txt_path = os.path.join(pair_path, 'sim3.txt')
        # 把最大生成树的结果备份
        run_str = 'cp ' + sim3_txt_path + ' ' + sim3_txt_path[:-4] + '_mst.txt'
        sim3_mat3x4 = value[0:3, : ]
        np.savetxt(sim3_txt_path, sim3_mat3x4)
    return

def read_sim3s_from_pose_graph(sim3_construction_path, sim3_tree_dict, suffix_str=''):
    sim3s_opted_txt_path = os.path.join(sim3_construction_path, 'sim3s_for_pose_graph_opted.txt')
    infile = open(sim3s_opted_txt_path, 'r')
    # 先得到根节点
    root_str = None
    sim3_tree_dict_spg = copy.deepcopy(sim3_tree_dict)
    for key, value in sim3_tree_dict.items():
        node_strs = key.split('+')
        root_str = node_strs[0]
        break
    # 读文件，检查该文件的name按照规则拼接后是否可以在最大生成树的字典里找到
    for line in infile.readlines():
        line = line.strip() # id name x y z q_x q_y q_z q_w s
        if line[0] == '#':
            continue
        strs = line.split(' ')
        name = strs[1]
        if name == root_str:
            continue
        tvec = [float(strs[2]), float(strs[3]), float(strs[4])]
        quat = [float(strs[8]), float(strs[5]), float(strs[6]), float(strs[7])]
        scale = float(strs[9])
        sim3_transform = compose_sim3_transform(scale, quat, tvec)
        key_str = root_str + '+' + name + suffix_str
        error_str = 'Error! ' + key_str + ' not in sim3_tree_dict_spg'
        assert key_str in sim3_tree_dict_spg, error_str
        sim3_tree_dict_spg[key_str] = sim3_transform
    return sim3_tree_dict_spg

def write_all_sim3_to_root_as_nodes(sim3_construction_path, node_id_dict, sim3_tree_dict, suffix_str=''):
    sim3_txt_path = os.path.join(sim3_construction_path, 'sim3s_for_pose_graph.txt')
    infile = open(sim3_txt_path, 'w')
    if not os.path.isdir(sim3_construction_path):
        print('Error path: ', sim3_construction_path) # 这是在最大生成树之后，如果路径还不存在，说明有问题
        exit(0)
    infile.write('# sim3 list with 1 line of data per sim3:\n')
    infile.write('# id name x y z q_x q_y q_z q_w s\n')
    root_str = None
    # 先写根节点的sim3
    for key, value in sim3_tree_dict.items():
        node_strs = key.split('+')
        root_str = node_strs[0]
        break
    line_str = str(node_id_dict[root_str]) + ' ' + root_str + ' 0.0 0.0 0.0 0.0 0.0 0.0 1.0 1.0\n'
    infile.write(line_str)
    # 再写其他节点的sim3
    for key, value in sim3_tree_dict.items():
        key_tmp = key[0:len(key)-len(suffix_str)]
        print(key, key_tmp)
        node_strs = key_tmp.split('+')
        leaf_str = node_strs[1]
        print('write_all_sim3_to_root_as_nodes key = ', key)
        print('leaf_str:', leaf_str)
        scale, quat, tvec = decompose_sim3_transform(value)
        line_str = str(node_id_dict[leaf_str]) + ' ' + leaf_str + ' ' + \
                    str(tvec[0]) + ' ' + str(tvec[1]) + ' ' + str(tvec[2]) + ' ' + \
                    str(quat[1]) + ' ' + str(quat[2]) + ' ' + str(quat[3]) + ' ' + \
                    str(quat[0]) + ' ' + str(scale) + '\n'
        infile.write(line_str)
    infile.close()
    return

def write_all_sim3_as_constraints(sim3_construction_path, node_constraint_list, suffix_str=''):
    sim3_txt_path = os.path.join(sim3_construction_path, 'sim3_constraints_for_pose_graph.txt')
    infile = open(sim3_txt_path, 'w')
    if not os.path.isdir(sim3_construction_path):
        print('Error path: ', sim3_construction_path) # 这是在最大生成树之后，如果路径还不存在，说明有问题
        exit(0)
    infile.write('# sim3_constraint list with 4 lines of data per constraint:\n')
    infile.write('# id_begin id_end\n')
    infile.write('# name_begin name_end\n')
    infile.write('# x y z q_x q_y q_z q_w s\n')
    infile.write('# information[0,0] information[1,1] ... information[6,6]\n')
    for constraint in node_constraint_list:
        # [node_id_dict[aa], node_id_dict[bb], aa, bb, sub_model_sim3_dict[aa_bb][2]]
        line_str = str(constraint[0]) + ' ' + str(constraint[1]) + '\n'
        infile.write(line_str)
        line_str = constraint[2] + ' ' + constraint[3] + '\n'
        infile.write(line_str)
        print('write_all_sim3_as_constraints line_str = ', line_str)
        scale, quat, tvec = decompose_sim3_transform(constraint[4])
        line_str = str(tvec[0]) + ' ' + str(tvec[1]) + ' ' + str(tvec[2]) + ' ' + \
                    str(quat[1]) + ' ' + str(quat[2]) + ' ' + str(quat[3]) + ' ' + \
                    str(quat[0]) + ' ' + str(scale) + '\n'
        infile.write(line_str)
        line_str = '0.0000001 0.0000001 0.0000001 0.0000001 0.0000001 0.0000001 0.0000001\n'
        infile.write(line_str)
    infile.close()
    return

def sim3merger_tree(db_path, model_path, sim3_construction_path, colmap_exe, sim3_tree_dict, max_node, suffix_str=''):
    # 用于存储中间文件
    sim3_construction_db_path = os.path.join(db_path, 'sim3_construction_db')
    sim3_construction_model_path = os.path.join(model_path, 'sim3_construction_model')
    if not os.path.isdir(sim3_construction_db_path):
        os.mkdir(sim3_construction_db_path)
    if not os.path.isdir(sim3_construction_model_path):
        os.mkdir(sim3_construction_model_path)
    # 增量式合并db和model
    is_first_merger = True
    database_path_loc = None
    database_path_map = None
    database_path_locmap = None
    model_folder_loc = None
    model_folder_map = None
    model_folder_locmap = None
    idx = 0
    for key, _ in sim3_tree_dict.items():
        # 确保需要的中间文件/文件夹都存在
        strs = key.split('+')
        leaf_str = strs[1]
        pair_path = os.path.join(sim3_construction_path, key)
        assert os.path.isdir(pair_path), 'Error! no folder ---> ' + pair_path
        # 处理db和model路径
        database_path_loc = os.path.join(db_path, leaf_str + '.db')
        model_folder_loc = os.path.join(model_path, leaf_str + '/0')
        if not is_first_merger:
            database_path_map = database_path_locmap
            model_folder_map = model_folder_locmap
        else:
            is_first_merger = False
            database_path_map = os.path.join(db_path, max_node + suffix_str + '.db')
            model_folder_map = os.path.join(model_path, max_node + suffix_str + '/0')
        # locmap都放在新建的临时文件夹
        database_path_locmap = os.path.join(sim3_construction_db_path, max_node  + '_tmp' + str(idx) + '.db')
        model_folder_locmap = os.path.join(sim3_construction_model_path, max_node  + '_tmp' + str(idx))
        if not os.path.isdir(model_folder_locmap):
            os.mkdir(model_folder_locmap)
        idx += 1
        #合并
        print('run_database_merger...')
        run_database_merger(colmap_exe, database_path_loc, database_path_map, database_path_locmap)
        print('update_loc_model_id_refer_to_locmap_database...')
        model_folder_new_loc = os.path.join(model_folder_loc, 'tmp')
        if not os.path.isdir(model_folder_new_loc):
            os.mkdir(model_folder_new_loc)
        update_loc_model_id_refer_to_locmap_database(model_folder_loc, database_path_locmap, model_folder_new_loc)
        print('run_model_sim3_merger...')
        run_model_sim3_merger(colmap_exe, model_folder_new_loc, model_folder_map, pair_path, model_folder_locmap)
        shutil.rmtree(model_folder_new_loc)
    return database_path_locmap, model_folder_locmap

def read_scenegraph(args):
    print('start scene graph parsing...\n')
    configPath = os.path.join(args.projPath, args.configDir)
    if args.validSeqsFromJson:
        sceneTopology = getSceneTopology(args.projPath, configPath=configPath, sceneGraphName=args.sceneGraph,
                                         validSeqsFromJson=args.validSeqsFromJson)
    else:
        sceneTopology = getSceneTopology(args.projPath, projName=args.projName, batchName=args.batchName,
                                         sceneGraphName=args.sceneGraph)

    seqList, routeList, routePairList = getTopologyCheckTasks(sceneTopology)
    only_name_list = getOnlyNamesFromNodesInfo([seqList, routeList, routePairList])
    full_seq_list = only_name_list[0]
    full_route_list = only_name_list[1]
    full_routepair_list = only_name_list[2]
    return full_route_list, full_routepair_list

def run_sim3_pose_graph(sim3_pose_graph_app_path, sim3_construction_path):
    #./sim3_pose_graph_optimize sim3s_path[.txt] sim3_constraints_path[.txt] sim3s_opted_path[.txt]
    sim3s_txt_path = os.path.join(sim3_construction_path, 'sim3s_for_pose_graph.txt')
    sim3_constraints_txt_path = os.path.join(sim3_construction_path, 'sim3_constraints_for_pose_graph.txt')
    sim3s_opted_txt_path = os.path.join(sim3_construction_path, 'sim3s_for_pose_graph_opted.txt')
    run_str = sim3_pose_graph_app_path + ' ' + sim3s_txt_path + ' ' + sim3_constraints_txt_path + ' ' + sim3s_opted_txt_path
    print(run_str)
    os.system(run_str)
    return

def sim3merger_scenegraph(dbDsPath, modelDsPath, sim3_construction_path, full_route_list, full_routepair_list, sub_model_sim3_dict,
                         imageListSuffix='',
                         colmapPath='colmap',
                         sim3_pose_graph_app_path=''):
    # 根据读取的信息，构建graph
    current_graph, node_id_dict, node_constraint_list = \
        construct_nx_graph_sim3(full_route_list, full_routepair_list, sub_model_sim3_dict, imageListSuffix)
    # 根据graph，生成"最大生成tree"
    max_tree, max_node = get_nx_tree_with_root(current_graph)
    # 把graph和tree都画出来
    visualize_graph_and_tree(modelDsPath, current_graph, max_tree, imageListSuffix)
    # 获取所有子节点到根节点的sim3变换
    sim3_tree_dict = get_all_sim3_to_root(max_tree, max_node, sub_model_sim3_dict, imageListSuffix)
    # 把sim3变换写到临时文件，作为colmap命令行的输入
    write_all_sim3_to_root(sim3_construction_path, sim3_tree_dict)
    # 把相对于root的sim3位姿，写到txt文件里，用于sim3_pose_graph优化
    write_all_sim3_to_root_as_nodes(sim3_construction_path, node_id_dict, sim3_tree_dict, imageListSuffix)
    write_all_sim3_as_constraints(sim3_construction_path, node_constraint_list)
    # 调sim3_pose_graph优化
    if sim3_pose_graph_app_path != '':
        run_sim3_pose_graph(sim3_pose_graph_app_path, sim3_construction_path)
        # 根据优化结果更新sim3_leaf2root
        sim3_tree_dict = read_sim3s_from_pose_graph(sim3_construction_path, sim3_tree_dict, imageListSuffix)
        update_sim3s_by_pose_graph(sim3_construction_path, sim3_tree_dict)
    # 循环所有非根节点，以根节点为identity，把所有非根节点变换merge到根节点
    database_path_locmap, model_folder_locmap = sim3merger_tree(dbDsPath, modelDsPath, sim3_construction_path, colmapPath, sim3_tree_dict, max_node, imageListSuffix)
    return database_path_locmap, model_folder_locmap

def run_sim3merger_scenegraph(args):
    # 路径拼接
    dbDsPath = os.path.join(args.projPath, args.dbDsDir)
    modelDsPath = os.path.join(args.projPath, args.modelDsDir)
    taskPath = os.path.join(args.projPath, args.taskDir)
    sim3_construction_path = os.path.join(taskPath, 'sim3_construction')

    # 读取scene_graph
    full_route_list, full_routepair_list = read_scenegraph(args)
    # 读取model状态和sim3信息
    sub_model_sim3_dict = read_model_and_sim3_txts(modelDsPath, full_route_list, full_routepair_list, args.imageListSuffix)
    # 根据读取的信息，构建graph
    current_graph, node_id_dict, node_constraint_list = \
        construct_nx_graph_sim3(full_route_list, full_routepair_list, sub_model_sim3_dict, args.imageListSuffix)
    # 根据graph，生成"最大生成tree"
    max_tree, max_node = get_nx_tree_with_root(current_graph)
    # 把graph和tree都画出来
    visualize_graph_and_tree(modelDsPath, current_graph, max_tree, args.imageListSuffix)
    # 获取所有子节点到根节点的sim3变换
    sim3_tree_dict = get_all_sim3_to_root(max_tree, max_node, sub_model_sim3_dict, args.imageListSuffix)
    # 把sim3变换写到临时文件，作为colmap命令行的输入
    write_all_sim3_to_root(sim3_construction_path, sim3_tree_dict)
    # 把相对于root的sim3位姿，写到txt文件里，用于sim3_pose_graph优化
    write_all_sim3_to_root_as_nodes(sim3_construction_path, node_id_dict, sim3_tree_dict, args.imageListSuffix)
    write_all_sim3_as_constraints(sim3_construction_path, node_constraint_list)
    # 调sim3_pose_graph优化
    if args.sim3_pose_graph_app_path != '':
        run_sim3_pose_graph(args.sim3_pose_graph_app_path, sim3_construction_path)
        # 根据优化结果更新sim3_leaf2root
        sim3_tree_dict = read_sim3s_from_pose_graph(sim3_construction_path, sim3_tree_dict, args.imageListSuffix)
        update_sim3s_by_pose_graph(sim3_construction_path, sim3_tree_dict)
    # 循环所有非根节点，以根节点为identity，把所有非根节点变换merge到根节点
    sim3merger_tree(dbDsPath, modelDsPath, sim3_construction_path, args.colmapPath, sim3_tree_dict, max_node, args.imageListSuffix)
    return

def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--projPath', required=True)
    parser.add_argument('--projName', type=str, default=None, help='如果未使用--validSeqsFromJson，则必须指定该参数')
    parser.add_argument('--batchName', type=str, default=None, help='如果未使用--validSeqsFromJson，则必须指定该参数')

    parser.add_argument('--dbDir', default='database')
    parser.add_argument('--modelDir', default='sparse')
    parser.add_argument('--dbDsDir', default='database_ds')
    parser.add_argument('--modelDsDir', default='sparse_ds')
    parser.add_argument('--imageListSuffix', default='', type=str)
    parser.add_argument('--configDir', default='config')
    parser.add_argument('--taskDir', default='tasks')
    parser.add_argument('--validSeqsFromJson', action='store_true')
    parser.add_argument('--sceneGraph', type=str, default='scene_graph')
    parser.add_argument('--sim3_pose_graph_app_path', default='', type=str)
    parser.add_argument('--colmapPath', default="colmap")

    args = parser.parse_args()

    return args

def main():
    args = parseArgs()
    run_sim3merger_scenegraph(args)

if __name__ == "__main__":
    main()