# coding: utf-8
import os
import sys
import argparse
import logging
import numpy as np
import networkx as nx # 图, 相关算法
import time
import matplotlib.pyplot as plt
import shutil
sys.path.append('../')

from colmap_process_loop_folders.videos_to_images_with_lists import extract_image_from_video
from colmap_process_loop_folders.tree_topology import getSceneTopology, getTopologyCheckTasks,\
    getTopologyCheckTasksSpecified, getOnlyNamesFromNodesInfo, filterTopologyCheckTasks, cleanSceneTopology,\
    writeSceneTolopogy, getFailedRoutesAndSeqs, setSeqCheckRunMapperFlags, filterTopologyCheckTasksByPlaces
from colmap_process_loop_folders.colmap_seq_sfms import run_colmap_seq_sfms
from colmap_process_loop_folders.colmap_kfs_selecter import run_colmap_kfs_selecter
from colmap_process_loop_folders.colmap_route_sfms import runAllRoutesSFM, runAllRoutePairsSFM
from colmap_process_loop_folders.colmap_models_analyzer import run_colmap_models_analyzer, read_model_reports
from log_file.py_logging import Logger
from colmap_process_loop_folders.basic_colmap_operation import checkExistence, cleanColmapMaterialsByNames
from log_file.logstdout import Logstdout

def write_rank_routes(modelDsPath, nodes_subgraph_idx, suffix_str2=''):
    report_path = modelDsPath + '/rank_routes' + suffix_str2 + '.txt'
    print('write to --->', report_path)
    geo_file = open(report_path,'w')
    geo_line = '# route_name rank current_degree expected_degree subgraph_centrality \n'
    print(geo_line)
    geo_file.write(geo_line)

    for key,value in nodes_subgraph_idx.items():
        # print(key, value)
        geo_line = key + ' ' + str(value[0]) + ' ' + str(value[1]) + ' ' + \
            str(value[2]) + ' ' + str(format(value[3], '.4f')) + '\n'
        geo_file.write(geo_line)

    geo_file.close()
    return

def rank_routes(scene_graph, current_graph):
    # subgraph_centrality: 使用邻接矩阵， 不支持weight， 求eigenvalue、eigenvector
    nodes_subgraph = nx.algorithms.centrality.subgraph_centrality(scene_graph)
    nodes_subgraph = dict(sorted(nodes_subgraph.items(), key=lambda item: -item[1]))
    # for key,value in nodes_subgraph.items():
    #     print(key, value)
    # print(' ')

    # 规则算法by wangcheng
    # 按照route-score排序
    # 如果route-score相同， expected_degree越大， 质量越差
    # 如果expected_degree也相同， 该route在scene_graph的理论权重越大， 质量越差
    nodes = scene_graph.nodes()
    nodes_route_score = {}
    for node in nodes:
        expected_degree = scene_graph.degree(node)
        current_degree = current_graph.degree(node)
        assert_str = 'Error! ' + node + ' has no edge, please check the scene_graph!'
        assert expected_degree > 0, assert_str
        route_score = float(current_degree) / float(expected_degree)
        nodes_route_score[node] = [route_score, current_degree, expected_degree, nodes_subgraph[node]]
    # 规则算法排序
    nodes_route_score = dict( sorted( nodes_route_score.items(), key=lambda item: ( item[1][0], -item[1][2], -item[1][3] ) ) )
    # for key, value in nodes_route_score.items():
    #     print(key, value)
    # print(' ')

    # 实际: 按照从差到好排序
    # 理论: 按照从好到差排序
    # 实际越差，理论越重要，越应该被重采
    # 把两组排序的排名（从0开始），相加，再排序，如果相加排名一致，按照理论重要程度排名
    # 存储subgraph的排序和得分
    nodes_subgraph_idx = nodes_route_score.copy()
    idx = 0
    for key, _ in nodes_subgraph_idx.items():
        # 把第一个值，从百分比，改成idx
        nodes_subgraph_idx[key][0] = idx
        idx += 1
    # 理论跟实际的排名相加
    idx = 0
    for key, _ in nodes_subgraph.items():
        nodes_subgraph_idx[key][0] += idx
        idx += 1
        
    # 按照相加排名排序
    # 如果相加排名相同，理论得分越高，越应该被重采
    nodes_subgraph_idx = dict(sorted( nodes_subgraph_idx.items(), key=lambda item: ( item[1][0], -item[1][3] ) ))
    # for key,value in nodes_subgraph_idx.items():
    #     print(key, value)

    return nodes_subgraph_idx

def construct_nx_graph(full_route_list, full_routepair_list, sub_model_status_dict, suffix_str=''):
    scene_graph = nx.Graph()
    current_graph = nx.Graph()
    # weight只是为了可视化布局，没有意义
    success_wight = 1
    unsuccess_wight = 0.8

    # 添加node， 每个node如果成功是绿色， 如果失败是红色
    for route_name in full_route_list:
        if sub_model_status_dict[route_name + suffix_str] == 'success':
            scene_graph.add_node(route_name, weight=success_wight, success=1, color='g')
            current_graph.add_node(route_name, color='g')
        else:
            scene_graph.add_node(route_name, weight=unsuccess_wight, success=0, color='r')
            current_graph.add_node(route_name, color='r')
        
    # 添加edge， 每条edge如果成功是绿色， 如果失败是红色
    for routepair_name in full_routepair_list:
        aa = routepair_name[0]
        bb = routepair_name[1]
        aa_bb = aa + '_' + bb + suffix_str
        if sub_model_status_dict[aa_bb] == 'success':
            scene_graph.add_edge(aa, bb, weight=success_wight, success=1, color='g')
            current_graph.add_edge(aa, bb)
        else:
            scene_graph.add_edge(aa, bb, weight=unsuccess_wight, success=0, color='r')
        
    return scene_graph, current_graph

def visualize_nx_graph(scene_graph, current_graph, modelDsPath, suffix_str2=''):
    # 不同的layout影响绘图时nodes的分布摆放
    # pos = nx.spectral_layout(scene_graph)
    # pos = nx.spring_layout(scene_graph, k=0.9, iterations=50, weight=None)
    pos = nx.kamada_kawai_layout(scene_graph, weight=None)
    x_values, y_values = zip(*pos.values())
    x_max = max(x_values)
    x_min = min(x_values)
    x_margin = (x_max - x_min) * 0.20
    
    nodes = scene_graph.nodes()
    edges = scene_graph.edges()
    node_colors = [scene_graph.nodes[node]['color']  for node in nodes]
    edge_colors = [scene_graph.edges[edge]['color']  for edge in edges]
    # 统一绘图
    nx.draw(scene_graph, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors, font_size=6)
    # plt.show()
    save_name = modelDsPath + '/graph_status' + suffix_str2 + '.png'
    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.savefig(save_name, dpi=1000)

    plt.close()
    current_nodes = current_graph.nodes()
    current_node_colors = [current_graph.nodes[node]['color']  for node in current_nodes]
    nx.draw(current_graph, pos, with_labels=True, node_color=current_node_colors, font_size=6)
    save_name = modelDsPath + '/graph_current' + suffix_str2 + '.png'
    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.savefig(save_name, dpi=1000)
    return

def statisticFrameNum(statisticalFile, imagesPath, fullSeqList, partSeqList, imageListSuffixValid):
    frameNumsDict = {}
    frameNumsFullSeq = 0
    frameNumsPartSeq = 0

    for seqName in fullSeqList:
        seqImageListFile = os.path.join(imagesPath, seqName + imageListSuffixValid + '.txt')
        frameNum = len(readStrLines(seqImageListFile))

        frameNumsFullSeq += frameNum

        if seqName in partSeqList:
            frameNumsPartSeq += frameNum
        
        frameNumsDict[seqName] = frameNum
    
    with open(statisticalFile, 'w') as fp:
        fp.write('frameNumsFullSeq = %d\n' % frameNumsFullSeq)
        fp.write('frameNum of each sequence:\n')
        for seqName in fullSeqList:
            fp.write('frameNum of %s: %d\n' % (seqName, frameNumsDict[seqName]))
        
        fp.write('\nframeNumsPartSeq = %d\n' % frameNumsPartSeq)
        fp.write('frameNum of each sequence:\n')
        for seqName in partSeqList:
            fp.write('frameNum of %s: %d\n' % (seqName, frameNumsDict[seqName]))

    return

def checkDirsAndCreate(dirList):
    for dirPath in dirList:
        if not os.path.isdir(dirPath):
            os.mkdir(dirPath)

    return

def readStrLines(file):
    lines = []
    with open(file) as fp:
        lines = fp.readlines()

    return lines

def getUniqueSuffix(inList):
    outList = []
    for ele in inList:
        if ele == '#':
            ele = ''

        if not ele in outList:
            outList.append(ele)

    return outList

def checkSuffixSetting(imageListSuffixes, imageListSuffixValid, seqToCheckList, imagesPath):
    if len(imageListSuffixes) == 0:
        imageListSuffixes.append(imageListSuffixValid)
    else:
        if (imageListSuffixValid in imageListSuffixes) and (len(imageListSuffixes) > 1):
            exceptStr = 'imageListSuffixValid in imageListSuffixes.' + \
                'This will take the raisk of overlapping the imageList with imageListSuffixValid.'
            raise Exception(exceptStr)
        if len(seqToCheckList)==0:
            exceptStr = 'You have set multiple suffix in imageListSuffixes, which means you want the script to determine ' + \
                'which suffix can make the route SFM success. However, there is no sequence in seqToCheckList.'
            raise Exception(exceptStr)

    checkImageListExistence(imageListSuffixes, seqToCheckList, imagesPath)

    return

def checkImageListExistence(imageListSuffixes, seqToCheckList, imagesPath):
    for suffix in imageListSuffixes:
        for seq in seqToCheckList:
            imageListFile = os.path.join(imagesPath, seq+suffix+'.txt')
            if not os.path.isfile(imageListFile):
                raise Exception("%s does not exist." % imageListFile)

    return

def copyImageListAsValidSuffix(imageListSuffixCur, imageListSuffixValid, seqToTryList, imagesPath):
    if not (imageListSuffixCur == imageListSuffixValid):
        for seq in seqToTryList:
            srcFile = os.path.join(imagesPath, seq + imageListSuffixCur + '.txt')
            dstFile = os.path.join(imagesPath, seq + imageListSuffixValid + '.txt')
            shutil.copy(srcFile, dstFile)

    return

def run_previous_topology_check(args):
    logStdoutFile = os.path.join(args.projPath, 'logStdout_previous_topology_check.txt')
    sys.stdout = Logstdout(logStdoutFile, sys.stdout)

    log_file_path = os.path.join(args.projPath, 'log_previous_topology_check.txt')
    logger = Logger(logfilename=log_file_path, logger="log_previous_topology_check").getlog()
    logger.info("start--->run_previous_topology_check")
    time_total_start = time.time()
    #---------------------------------------------------------------------------
    imagesDsPath = os.path.join(args.projPath, args.imagesDsDir)
    configPath = os.path.join(args.projPath, args.configDir)

    #---------------------------------------------------------------------------
    # build scene topology and get task lists for topology check
    # 总文件列表的入口，文件IO如果出错，首先debug这里
    logger.info("start scene graph parsing...")
    time_start = time.time()

    # if there exists suppleGraphFile, running with map supplement mode
    mapSuppleMode = False
    suppleGraphFile = os.path.join(args.projPath, args.supplementalGraph+'.json')
    if os.path.isfile(suppleGraphFile):
        mapSuppleMode = True

    supplePlaces = []
    if args.validSeqsFromJson:
        sceneTopology = getSceneTopology(args.projPath, configPath=configPath, sceneGraphName=args.sceneGraph,
                                         validSeqsFromJson=args.validSeqsFromJson,
                                         suppleGraphName=args.supplementalGraph,
                                         supplePlaces=supplePlaces,
                                         imagesDir=args.imagesDsDir)
    else:
        sceneTopology = getSceneTopology(args.projPath, projName=args.projName, batchName=args.batchName,
                                         sceneGraphName=args.sceneGraph,
                                         suppleGraphName=args.supplementalGraph,
                                         supplePlaces=supplePlaces,
                                         imagesDir=args.imagesDsDir)

    seqList, routeList, routePairList = getTopologyCheckTasks(sceneTopology)
    only_name_list = getOnlyNamesFromNodesInfo([seqList, routeList, routePairList])
    full_seq_list = only_name_list[0]
    full_route_list = only_name_list[1]
    full_routepair_list = only_name_list[2]

    if not (args.tpCheckList==None):
        if os.path.exists(args.tpCheckList):
            strLines = readStrLines(args.tpCheckList)
            seqToCheckList, routeToCheckList, routePairToCheckList = \
                getTopologyCheckTasksSpecified(strLines, sceneTopology)
        else:
            logger.error('Specified tpCheckListFile does not exist: %s\n' % args.tpCheckList)
            raise Exception('Specified tpCheckListFile does not exist: %s\n' % args.tpCheckList)
    else:
        if mapSuppleMode:
            seqToCheckList, routeToCheckList, routePairToCheckList = \
                filterTopologyCheckTasksByPlaces(seqList, routeList, routePairList, sceneTopology, supplePlaces)
        else:
            seqToCheckList = seqList.copy()
            routeToCheckList = routeList.copy()
            routePairToCheckList = routePairList.copy()

    only_name_list = getOnlyNamesFromNodesInfo([seqToCheckList, routeToCheckList, routePairToCheckList])
    part_seq_list = only_name_list[0]
    part_route_list = only_name_list[1]
    part_routepair_list = only_name_list[2]
    time_cost = time.time() - time_start
    logger.info("<parse scene graph>time_cost = " + str(time_cost) + "s")

    # check existentce of sequence imageDir
    checkExistence(part_seq_list, imagesDsPath)

    #---------------------------------------------------------------------------
    dbDsPath = os.path.join(args.projPath, args.dbDsDir)
    modelDsPath = os.path.join(args.projPath, args.modelDsDir)
    checkDirsAndCreate([dbDsPath, modelDsPath])

    # try each imageListSuffix in args.imageListSuffixTry
    imageListSuffixes = getUniqueSuffix(args.imageListSuffixTry)
    imageListSuffixValid = args.imageListSuffix
    checkSuffixSetting(imageListSuffixes, imageListSuffixValid, part_seq_list, imagesDsPath)

    triedSuffixCnt = 0
    seqToTryList = part_seq_list.copy()
    routeToTryList = routeToCheckList.copy()
    seqRunMapperFlags = setSeqCheckRunMapperFlags(seqToTryList, sceneTopology)
    route_status_dict = None
    while (triedSuffixCnt<len(imageListSuffixes)) and ((len(seqToTryList)>0) or (len(routeToTryList)>0)):
        imageListSuffixCur = imageListSuffixes[triedSuffixCnt]
        copyImageListAsValidSuffix(imageListSuffixCur, imageListSuffixValid, seqToTryList, imagesDsPath)

        # sequence SFM
        logger.info("start sequence check...")
        time_start = time.time()
        
        # 现在只关心route的成败，所以不需要做seq-sfm，只需要做seq-match，optionalMapper=True即可
        run_colmap_seq_sfms(args.colmapPath, imagesDsPath, dbDsPath, modelDsPath, seqToTryList, 
                            seq_run_mapper_flags = seqRunMapperFlags,
                            suffix_str=imageListSuffixValid,
                            seq_match_overlap=20, mapper_min_num_matches=50,
                            mapper_init_min_num_inliers=200, mapper_abs_pose_min_num_inliers=60,
                            optionalMapper=True)
        # models analyzer
        # optionalMapper表示run_colmap_seq_sfms只跑特征提取和序列匹配，不跑mapper，所以这里的分析就不用做了
        # seq_status_dict = run_colmap_models_analyzer(dbDsPath, modelDsPath, full_seq_list, suffix_str=imageListSuffixValid, suffix_str2='_seq',
        #             mean_obs_per_img_threshold=150.0, mean_track_length_threshold=3.0,
        #             valid_img_ratio_threshold=0.8, mean_rep_error_threshold=4.0)
        time_cost = time.time() - time_start
        logger.info("<sequence check>time_cost = " + str(time_cost) + "s")
        
        #---------------------------------------------------------------------------
        # route SFM
        logger.info("start route check...")
        time_start = time.time()
        runAllRoutesSFM(args.colmapPath, imagesDsPath, dbDsPath, modelDsPath, routeToTryList,
                        mapper_min_num_matches=50, mapper_init_min_num_inliers=200, imageListSuffix=imageListSuffixValid)
        # models analyzer
        route_status_dict = run_colmap_models_analyzer(dbDsPath, modelDsPath, full_route_list, suffix_str=imageListSuffixValid, suffix_str2='_route',
                    mean_obs_per_img_threshold=150.0, mean_track_length_threshold=3.0,
                    valid_img_ratio_threshold=0.8, mean_rep_error_threshold=4.0)
        time_cost = time.time() - time_start
        logger.info("<route check>time_cost = " + str(time_cost) + "s")

        routeToTryList, seqToTryList = getFailedRoutesAndSeqs(routeToTryList, route_status_dict, sceneTopology)

        triedSuffixCnt += 1

    # 统计所有sequence的有效帧数
    statisticalFile = os.path.join(modelDsPath, 'frameNumStatistics.txt')
    statisticFrameNum(statisticalFile, imagesDsPath, full_seq_list, part_seq_list, imageListSuffixValid)

    #---------------------------------------------------------------------------
    # routePair SFM
    if route_status_dict == None:
        route_status_dict = run_colmap_models_analyzer(dbDsPath, modelDsPath, full_route_list, suffix_str=imageListSuffixValid, suffix_str2='_route',
                    mean_obs_per_img_threshold=150.0, mean_track_length_threshold=3.0,
                    valid_img_ratio_threshold=0.8, mean_rep_error_threshold=4.0)

    logger.info("start route pair check...")
    time_start = time.time()
    mean_rep_error_th = 4.0
    valid_img_ratio_th = 0.8
    if args.useMergedSFMPlus:
        mean_rep_error_th = 16.0
        valid_img_ratio_th = 0.99 # 如果merge成功，那么有效的图像百分比应该是100%

    routePairToCheckList, deleteTaskNameList = filterTopologyCheckTasks(routePairToCheckList, route_status_dict, sceneTopology)
    cleanColmapMaterialsByNames(deleteTaskNameList, imagesDsPath, dbDsPath, modelDsPath, imageListSuffix=imageListSuffixValid)

    runAllRoutePairsSFM(args.colmapPath, imagesDsPath, dbDsPath, modelDsPath, routePairToCheckList,
                        mapper_min_num_matches=50, mapper_init_min_num_inliers=200, imageListSuffix=imageListSuffixValid,
                        useMergedSFMPlus=args.useMergedSFMPlus,
                        merger_max_reproj_error=mean_rep_error_th, merger_min_2d_inlier_percent=0.8,
                        mapper_fix_existing_images=1)
    # models analyzer
    routepair_status_dict = run_colmap_models_analyzer(dbDsPath, modelDsPath, full_routepair_list, suffix_str=imageListSuffixValid,  suffix_str2='_routepair',
                mean_obs_per_img_threshold=150.0, mean_track_length_threshold=3.0,
                valid_img_ratio_threshold=valid_img_ratio_th, mean_rep_error_threshold=mean_rep_error_th)
    time_cost = time.time() - time_start
    logger.info("<route pair check>time_cost = " + str(time_cost) + "s")
    # ---------------------------------------------------------------------------
    # clean sceneTopology and write to scene_graph_checked.json
    logger.info("start clean sceneTopology...")
    time_start = time.time()
    sceneTopologyChecked = cleanSceneTopology(route_status_dict, routepair_status_dict, routeList, routePairList, sceneTopology)
    if mapSuppleMode:
        sceneGraphChecked = os.path.join(args.projPath, args.sceneGraph + '_' + args.supplementalGraph + '_checked' + imageListSuffixValid + '.json')
    else:
        sceneGraphChecked = os.path.join(args.projPath, args.sceneGraph+'_checked' + imageListSuffixValid + '.json')
        
    writeSceneTolopogy(sceneTopologyChecked, sceneGraphChecked)

    #---------------------------------------------------------------------------
    # scene_graph GLOBAL check
    sub_model_status_dict = read_model_reports(modelDsPath, full_route_list, full_routepair_list, imageListSuffixValid)
    scene_graph, current_graph = construct_nx_graph(full_route_list, full_routepair_list, sub_model_status_dict, imageListSuffixValid)
    visualize_nx_graph(scene_graph, current_graph, modelDsPath, suffix_str2=imageListSuffixValid)
    nodes_subgraph_idx = rank_routes(scene_graph, current_graph)
    write_rank_routes(modelDsPath, nodes_subgraph_idx, suffix_str2=imageListSuffixValid)
    time_cost = time.time() - time_start
    logger.info("<clean sceneTopology>time_cost = " + str(time_cost) + "s")
    time_total_cost = time.time() - time_total_start
    logger.info("<time_total_cost>time_total_cost = " + str(time_total_cost) + "s")
    logger.info("done.")

def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--projPath', required=True)
    parser.add_argument('--projName', type=str, default=None, help='如果使用了--validSeqsFromJson，则必须指定该参数')
    parser.add_argument('--batchName', type=str, default=None, help='如果使用了--validSeqsFromJson，则必须指定该参数')

    parser.add_argument('--imagesDsDir', default='images_ds')
    parser.add_argument('--dbDsDir', default='database_ds')
    parser.add_argument('--modelDsDir', default='sparse_ds')
    parser.add_argument('--imageListSuffixTry', type=str, nargs = '+', default=[])
    parser.add_argument('--imageListSuffix', type=str, default='')
    parser.add_argument('--configDir', default='config')
    parser.add_argument('--taskDir', default='tasks')
    parser.add_argument('--validSeqsFromJson', action='store_false')
    parser.add_argument('--sceneGraph', type=str, default='scene_graph')
    parser.add_argument('--supplementalGraph', type=str, default='')

    parser.add_argument('--colmapPath', default="colmap")

    parser.add_argument('--tpCheckList', default=None, type=str)
    parser.add_argument('--useMergedSFMPlus', action='store_true')

    args = parser.parse_args()

    return args

def main():
    args = parseArgs()
    run_previous_topology_check(args)

if __name__ == "__main__":
    main()