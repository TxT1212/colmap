# coding: utf-8
import os, sys
scriptPath = sys.path[0]
sys.path.append(os.path.dirname(scriptPath))
import argparse
import logging
import time
import colmap_process_loop_folders.tree_topology as tt
import colmap_process_loop_folders.basic_colmap_operation as bco
from colmap_process_loop_folders.colmap_seq_sfms import run_colmap_seq_sfms
from colmap_process.colmap_seq_sfm import run_custom_sfm, run_image_registrator, run_model_merger, run_custom_matcher, \
     run_model_merger_plus, run_exhaustive_sfm, run_model_aligner, run_point_triangulator, run_bundle_adjuster, run_mapper, \
     run_feature_extractor, run_exhaustive_matcher
from colmap_process.colmap_model_rescale import rescale_model
from colmap_process.colmap_model_modify import update_loc_model_id_refer_to_locmap_database
from python_scale_ezxr.run_scale_ezxr import get_charuco_scale
from colmap_sim3merger_scenegraph import sim3merger_scenegraph, read_sim3_txt
from colmap_process.colmap_export_geo import colmap_export_geo
import shutil
from log_file.py_logging import Logger
from colmap_process.colmap_map_merger_plus import run_submodel_merger
from colmap_process.colmap_get_submodel import run_image_deleter
from log_file.logstdout import Logstdout

def getSubNodeMaterials(subNodeNamesInDict, imagesPath, dbPath, imageListSuffix=''):
    subListFiles = []
    subDBFiles = []
    for subNodeNameInDict in subNodeNamesInDict:
        subImageListFile = os.path.join(imagesPath, subNodeNameInDict + imageListSuffix + '.txt')
        subListFiles.append(subImageListFile)

        subDBFile = os.path.join(dbPath, subNodeNameInDict + imageListSuffix + '.db')
        if os.path.isfile(subDBFile):
            subDBFiles.append(subDBFile)
    
    return subListFiles, subDBFiles

def augmentSFM(augNode, parentNode, mapNodes, imagesPath, dbPath, modelPath,
               colmapPath="colmap",
               matcher_min_num_inliers=15,
               mapper_min_num_matches=45,
               mapper_init_min_num_inliers=200,
               mapper_abs_pose_min_num_inliers=60,
               skip_feature_extractor=True,
               apply_model_aligner=False,
               imageListSuffix='',
               mask_path='',
               mapper_fix_existing_images=0):
    if len(augNode['augEdges']) == 0:
        print('empty augEdges in %s, skip this augnode\n' % augNode['augName'])
        return []

    # if augDBFile exists, skip this augNode
    augImageListFile = os.path.join(imagesPath, augNode['augName'] + imageListSuffix + '.txt')
    augDBFile = os.path.join(dbPath, augNode['augName'] + imageListSuffix + '.db')
    augNodeModelPath = os.path.join(modelPath, augNode['augName'] + imageListSuffix)

    # get augmented routes and routePairs
    if parentNode.type == 'scene':
        assocRoutePairs = tt.getSceneHooksByEdge(parentNode.name, augNode['augEdges'], mapNodes)
    else:
        raise Exception('unsupported node type for now.')
    
    assocValidSeqsGraph = getValidSeqsGraphOfRoutePairs(assocRoutePairs, mapNodes)
    assocValidSeqsGraphInterPlaces = assocValidSeqsGraph.copy()

    if os.path.isfile(augDBFile):
        print('%s already exists, skip this augnode\n' % augDBFile)
        return assocValidSeqsGraphInterPlaces

    augRoutes = []
    augSeqs = []
    augRoutesValidSeqsGraph = []
    for routePairs in assocRoutePairs:
        for route in routePairs:
            routeNode = mapNodes[route]
            if (not (routeNode.parent == augNode['baseName'])) and (not (route in augRoutes)):
                augRoutes.append(route)
                augSeqs += routeNode.validSeqs
                augRoutesValidSeqsGraph += routeNode.validSeqsGraph

    # merge imageList, db
    subNodeNamesInDict = [augNode['baseName']] + augRoutes
    subListFiles , subDBFiles = getSubNodeMaterials(subNodeNamesInDict, imagesPath, dbPath, imageListSuffix=imageListSuffix)
    
    hasAllSeqsDB = True
    if len(subDBFiles) < len(subNodeNamesInDict):
        subNodeNamesInDict = [augNode['baseName']] + augSeqs
        subListFiles , subDBFiles = getSubNodeMaterials(subNodeNamesInDict, imagesPath, dbPath, imageListSuffix=imageListSuffix)
        assocValidSeqsGraph += augRoutesValidSeqsGraph

        if len(subDBFiles) < len(subNodeNamesInDict):
            hasAllSeqsDB = False
            subDBFiles=subDBFiles[0:1]
            assocValidSeqsGraph += [list(t) for t in list(zip(augSeqs, augSeqs.copy()))]

    bco.mergeImageListFile(subListFiles, augImageListFile)
    bco.mergeDBFile(subDBFiles, augDBFile, colmapPath=colmapPath)

    # create matchList
    augMatchListFile = os.path.join(dbPath, augNode['augName'] + imageListSuffix + '_match.txt')
    bco.createMatchList(imagesPath, assocValidSeqsGraph, augMatchListFile, imageListSuffix=imageListSuffix)

    # run colmapSFM
    if os.path.isdir(augNodeModelPath):
        shutil.rmtree(augNodeModelPath)
    os.mkdir(augNodeModelPath)

    inputModelPath = bco.getOneValidModelPath([os.path.join(modelPath, augNode['baseName'] + imageListSuffix)])

    run_custom_sfm(colmapPath, augDBFile, imagesPath, augImageListFile, augMatchListFile, augNodeModelPath,
                   matcher_min_num_inliers=matcher_min_num_inliers, mapper_min_num_matches=mapper_min_num_matches,
                   mapper_init_min_num_inliers=mapper_init_min_num_inliers,
                   mapper_abs_pose_min_num_inliers=mapper_abs_pose_min_num_inliers,
                   mapper_input_model_path=inputModelPath,
                   skip_feature_extractor=(skip_feature_extractor and hasAllSeqsDB),
                   apply_model_aligner=apply_model_aligner,
                   mask_path=mask_path,
                   mapper_fix_existing_images=mapper_fix_existing_images)

    return assocValidSeqsGraphInterPlaces

def pairwiseModelAlign(augNodes, modelPath, imageListSuffix='', colmapPath="colmap", useSubModelMerger=True, sim3MaxReprojError=16):
    nodeList = []
    nodePairList = []
    sim3Dict = {}
    
    modelStatus = 'success'
    for augNode in augNodes:
        nodeName = augNode['baseName']
        nodeList.append(nodeName)
        sim3Dict[nodeName+imageListSuffix] = [modelStatus, None, None]

        for linkedNode in augNode['augNodes']:
            nodePairList.append([nodeName, linkedNode])

            alignedModelPath = os.path.join(modelPath, linkedNode+'_alignedTo_'+augNode['augName'] + imageListSuffix, '0')
            if os.path.isdir(os.path.dirname(alignedModelPath)):
                shutil.rmtree(os.path.dirname(alignedModelPath))
            os.makedirs(alignedModelPath)

            linkedNodeModelPath = bco.getOneValidModelPath([os.path.join(modelPath, linkedNode + imageListSuffix)])
            augedNodeModelPath = bco.getOneValidModelPath([os.path.join(modelPath, augNode['augName'] + imageListSuffix)])

            if useSubModelMerger:
                run_submodel_merger(colmapPath, linkedNodeModelPath, augedNodeModelPath, alignedModelPath, sim3MaxReprojError)
                inlierNumber, sim3Transform = read_sim3_txt(os.path.join(alignedModelPath, 'mergered_submodel', 'sim3.txt'))
            else:
                run_model_aligner(colmapPath, linkedNodeModelPath, os.path.join(augedNodeModelPath, 'geos.txt'),
                                alignedModelPath, max_error=0.05)
                inlierNumber, sim3Transform = read_sim3_txt(os.path.join(alignedModelPath, 'sim3.txt'))

            sim3Dict[nodeName+'_'+linkedNode+imageListSuffix] = [modelStatus, inlierNumber, sim3Transform]

    return nodeList, nodePairList, sim3Dict

def sim3GroupedSFM(nodeNameInDict, subNodeNamesInDict, mapNodes, imagesPath, dbPath, modelPath, removeTmpData=True,
                   colmapPath="colmap",
                   matcher_min_num_inliers=15,
                   mapper_min_num_matches=45,
                   mapper_init_min_num_inliers=200,
                   mapper_abs_pose_min_num_inliers=60,
                   mapper_fix_existing_images=0,
                   imageListSuffix='', 
                   mask_path='',
                   sim3PoseGraphApp='',
                   matchMethods=['custom']):    
    # split subNodes and edges
    mapNode = mapNodes[nodeNameInDict]
    augmentedSubNodes = tt.splitGraph(mapNode.subNodes, mapNode.edges)

    splitResultsFile = os.path.join(dbPath, nodeNameInDict + imageListSuffix + '_splitInfo.txt')
    with open(splitResultsFile, 'w') as fp:
        for augSubNode in augmentedSubNodes:
            fp.write('%s\n' % augSubNode['augName'])

    # augment sfm
    augedSeqsGraph = []
    for augSubNode in augmentedSubNodes:
        augedSeqsGraph += augmentSFM(augSubNode, mapNode, mapNodes, imagesPath, dbPath, modelPath,
                                     colmapPath=colmapPath,
                                     matcher_min_num_inliers=matcher_min_num_inliers,
                                     mapper_min_num_matches=mapper_min_num_matches,
                                     mapper_init_min_num_inliers=mapper_init_min_num_inliers,
                                     mapper_abs_pose_min_num_inliers=mapper_abs_pose_min_num_inliers,
                                     skip_feature_extractor=True,
                                     apply_model_aligner=True,
                                     imageListSuffix=imageListSuffix,
                                     mask_path=mask_path,
                                     mapper_fix_existing_images=mapper_fix_existing_images)

    # pairwise sim3 calculation
    # get full_route_list, full_routepair_list, sub_model_sim3_dict for sim3merger_scenegraph
    subNodeList, subNodePairList, sim3Dict = pairwiseModelAlign(augmentedSubNodes, modelPath, 
                                                                imageListSuffix=imageListSuffix, 
                                                                colmapPath=colmapPath)
    
    # mst-based sfm merger (db merge -> loc_model_id_update -> sim3_model_merger) and poseGraph-based optimization
    sim3ConstructionPath = os.path.join(modelPath, 'sim3_construction')
    sim3ConstructionDBPath = os.path.join(dbPath, 'sim3_construction_db')
    sim3ConstructionModelPath = os.path.join(modelPath, 'sim3_construction_model')
    mergedDBPath, mergedModelPath = sim3merger_scenegraph(dbPath, modelPath, sim3ConstructionPath, subNodeList, subNodePairList, sim3Dict,
                                                          imageListSuffix=imageListSuffix,
                                                          colmapPath=colmapPath,
                                                          sim3_pose_graph_app_path=sim3PoseGraphApp)

    # copy merged data as mapNode data, then delete tmpData: sim3ConstructionPath, sim3ConstructionDBPath, sim3ConstructionModelPath
    imageListFile = os.path.join(imagesPath, nodeNameInDict + imageListSuffix + '.txt')
    subListFiles = []
    for subNode in subNodeList:
        subImageListFile = os.path.join(imagesPath, subNode + imageListSuffix + '.txt')
        subListFiles.append(subImageListFile)

    bco.mergeImageListFile(subListFiles, imageListFile)

    dbFile = os.path.join(dbPath, nodeNameInDict + imageListSuffix + '.db')
    nodeModelPath = os.path.join(modelPath, nodeNameInDict + imageListSuffix)
    nodeModelPathLeaf = os.path.join(nodeModelPath, '0')
    nodeModelPathJustMerge = os.path.join(modelPath, nodeNameInDict+ '_JustMerge' + imageListSuffix)
    nodeModelPathJustMergeLeaf = os.path.join(nodeModelPathJustMerge, '0')

    if os.path.isdir(nodeModelPath):
        shutil.rmtree(nodeModelPath)
    os.makedirs(nodeModelPathLeaf)

    if os.path.isdir(nodeModelPathJustMerge):
        shutil.rmtree(nodeModelPathJustMerge)
    os.makedirs(nodeModelPathJustMergeLeaf)

    shutil.copy(mergedDBPath, dbFile)
    bco.copyModelFiles(mergedModelPath, nodeModelPathJustMergeLeaf)
    colmap_export_geo(nodeModelPathJustMergeLeaf, [1,0,0,0,1,0,0,0,1])

    if removeTmpData:
        bco.checkAndDeleteFiles([sim3ConstructionPath, sim3ConstructionDBPath, sim3ConstructionModelPath])

    # augment match (exhaustive, spatial, custom)
    if 'custom' in matchMethods:
        matchListFile = os.path.join(dbPath, nodeNameInDict + imageListSuffix + '_match.txt')
        bco.createMatchList(imagesPath, augedSeqsGraph, matchListFile, imageListSuffix=imageListSuffix)

        # run custom match
        run_custom_matcher(colmapPath, dbFile, matchListFile, min_num_inliers=matcher_min_num_inliers)
    
    if True:
        # run mapper
        run_custom_sfm(colmapPath, dbFile, imagesPath, imageListFile, matchListFile, nodeModelPath,
                   matcher_min_num_inliers=matcher_min_num_inliers, mapper_min_num_matches=mapper_min_num_matches,
                   mapper_init_min_num_inliers=mapper_init_min_num_inliers,
                   mapper_abs_pose_min_num_inliers=mapper_abs_pose_min_num_inliers,
                   mapper_input_model_path=nodeModelPathJustMergeLeaf,
                   skip_feature_extractor=True,
                   skip_custom_match=True,
                   apply_model_aligner=True,
                   mask_path=mask_path,
                   mapper_fix_existing_images=mapper_fix_existing_images)
    else:
        # point_triangulator
        run_point_triangulator(colmapPath, dbFile, imagesPath, nodeModelPathJustMergeLeaf, nodeModelPathLeaf)

        # global ba
        run_bundle_adjuster(colmapPath, nodeModelPathLeaf, nodeModelPathLeaf)
        colmap_export_geo(nodeModelPathLeaf, [1,0,0,0,1,0,0,0,1])

    # global glued ba (optional)
    # colmap_export_geo(nodeModelPathLeaf, [1,0,0,0,1,0,0,0,1])

    return

def globalExhaustiveSFM(nodeNameInDict, mapNodes, imagesPath, dbPath, modelPath, colmapPath="colmap",
                     matcher_min_num_inliers=15,
                     mapper_min_num_matches=45,
                     mapper_init_min_num_inliers=200,
                     mapper_abs_pose_min_num_inliers=60,
                     skip_feature_extractor=False,
                     apply_model_aligner=False,
                     imageListSuffix='',
                     mask_path='',
                     useExhaustiveMatcher=True):
    mapNode = mapNodes[nodeNameInDict]
    validSeqs = mapNode.validSeqs

    # merge imageList
    imageListFile = os.path.join(imagesPath, nodeNameInDict + imageListSuffix + '.txt')
    dbFile = os.path.join(dbPath, nodeNameInDict + imageListSuffix + '.db')
    nodeModelPath = os.path.join(modelPath, nodeNameInDict + imageListSuffix)

    if os.path.isdir(nodeModelPath):
        shutil.rmtree(nodeModelPath)
    os.mkdir(nodeModelPath)

    if os.path.isfile(dbFile):
        os.remove(dbFile)

    # generage validSeqsGraph
    validSeqsGraph = []
    ordinarySeqs, charucoSeqsList = tt.getOrdinaryAndCharucoSeqs(mapNode, mapNodes)
    validSeqsGraphExceptOrdinary, validSeqsGraph = bco.matchOrdinaryAndCharucoSeqs(ordinarySeqs, charucoSeqsList)

    subListFiles = []
    subListFilesOrdinary = []
    for seqName in validSeqs:
        subImageListFile = os.path.join(imagesPath, seqName + imageListSuffix + '.txt')
        subListFiles.append(subImageListFile)

        if seqName in ordinarySeqs:
            subListFilesOrdinary.append(subImageListFile)

    bco.mergeImageListFile(subListFiles, imageListFile)

    # extractor and exhaustive match for ordinary seqs
    if useExhaustiveMatcher:
        imageListFileOrdinary = os.path.join(imagesPath, nodeNameInDict + '_ordinary' + imageListSuffix + '.txt')
        bco.mergeImageListFile(subListFilesOrdinary, imageListFileOrdinary)

        run_feature_extractor(colmapPath, dbFile, imagesPath, imageListFileOrdinary, mask_path=mask_path)
        run_exhaustive_matcher(colmapPath, dbFile, matcher_min_num_inliers)

        validSeqsGraph = validSeqsGraphExceptOrdinary

    # create matchList
    matchListFile = os.path.join(dbPath, nodeNameInDict + imageListSuffix + '_match.txt')
    bco.createMatchList(imagesPath, validSeqsGraph, matchListFile, imageListSuffix=imageListSuffix)

    # run colmapSFM
    run_custom_sfm(colmapPath, dbFile, imagesPath, imageListFile, matchListFile, nodeModelPath,
                  matcher_min_num_inliers=matcher_min_num_inliers, mapper_min_num_matches=mapper_min_num_matches,
                  mapper_init_min_num_inliers=mapper_init_min_num_inliers,
                  mapper_abs_pose_min_num_inliers=mapper_abs_pose_min_num_inliers,
                  mapper_input_model_path='',
                  skip_feature_extractor=skip_feature_extractor,
                  apply_model_aligner=apply_model_aligner,
                  mask_path=mask_path)

    return

def globalGroupedSFM(nodeNameInDict, subNodeNamesInDict, mapNodes, imagesPath, dbPath, modelPath, colmapPath="colmap",
                     matcher_min_num_inliers=15,
                     mapper_min_num_matches=45,
                     mapper_init_min_num_inliers=200,
                     mapper_abs_pose_min_num_inliers=60,
                     skip_feature_extractor=True,
                     apply_model_aligner=False,
                     imageListSuffix='',
                     mask_path='',
                     optionalMapper=False):
    mapNode = mapNodes[nodeNameInDict]

    # merge imageList and db
    imageListFile = os.path.join(imagesPath, nodeNameInDict + imageListSuffix + '.txt')
    dbFile = os.path.join(dbPath, nodeNameInDict + imageListSuffix + '.db')
    nodeModelPath = os.path.join(modelPath, nodeNameInDict + imageListSuffix)

    subListFiles = []
    subDBFiles = []
    subModelPaths = []
    for subNodeNameInDict in subNodeNamesInDict:
        subImageListFile = os.path.join(imagesPath, subNodeNameInDict + imageListSuffix + '.txt')
        subListFiles.append(subImageListFile)

        subDBFile = os.path.join(dbPath, subNodeNameInDict + imageListSuffix + '.db')
        subDBFiles.append(subDBFile)

        subModelPath = os.path.join(modelPath, subNodeNameInDict + imageListSuffix)
        subModelPaths.append(subModelPath)

    bco.mergeImageListFile(subListFiles, imageListFile)
    bco.mergeDBFile(subDBFiles, dbFile, colmapPath=colmapPath)

    # generage validSeqsGraph
    validSeqsGraph = []
    if mapNode.type == "route":
        validSeqsGraph = mapNode.validSeqsGraph
    else:
        raise Exception("unsupported node type for now.")

    # create matchList
    matchListFile = os.path.join(dbPath, nodeNameInDict + imageListSuffix + '_match.txt')
    bco.createMatchList(imagesPath, validSeqsGraph, matchListFile, imageListSuffix=imageListSuffix)

    # run colmapSFM
    run_custom_sfm(colmapPath, dbFile, imagesPath, imageListFile, matchListFile, nodeModelPath,
                  matcher_min_num_inliers=matcher_min_num_inliers, mapper_min_num_matches=mapper_min_num_matches,
                  mapper_init_min_num_inliers=mapper_init_min_num_inliers,
                  mapper_abs_pose_min_num_inliers=mapper_abs_pose_min_num_inliers,
                  mapper_input_model_path='',
                  skip_feature_extractor=skip_feature_extractor,
                  apply_model_aligner=apply_model_aligner,
                  mask_path=mask_path,
                  optionalMapper=optionalMapper)

    return

def mergedSFMPlus(localName, refName, mergedName, mergedValidSeqsGraph, imagesPath, dbPath, modelPath, colmapPath="colmap",
              matcher_min_num_inliers=15,
              mapper_min_num_matches=45,
              mapper_init_min_num_inliers=200,
              mapper_abs_pose_min_num_inliers=60,
              skip_feature_extractor=True,
              apply_model_aligner=False,
              imageListSuffix='',
              mask_path='',
              merger_max_reproj_error=64):
    time.sleep(2)
    # merge imageList and db
    imageListFile = os.path.join(imagesPath, mergedName + imageListSuffix + '.txt')
    dbFile = os.path.join(dbPath, mergedName + imageListSuffix + '.db')
    mergedModelPath = os.path.join(modelPath, mergedName + imageListSuffix)
    if os.path.isdir(mergedModelPath):
        shutil.rmtree(mergedModelPath)
    os.mkdir(mergedModelPath)

    subNodeNamesInDict = [refName, localName]
    subListFiles = []
    subDBFiles = []
    for subNodeNameInDict in subNodeNamesInDict:
        subImageListFile = os.path.join(imagesPath, subNodeNameInDict + imageListSuffix + '.txt')
        subListFiles.append(subImageListFile)

        subDBFile = os.path.join(dbPath, subNodeNameInDict + imageListSuffix + '.db')
        subDBFiles.append(subDBFile)

    bco.mergeImageListFile(subListFiles, imageListFile)
    bco.mergeDBFile(subDBFiles, dbFile, colmapPath=colmapPath)

    # create matchList
    matchListFile = os.path.join(dbPath, mergedName + imageListSuffix + '_match.txt')
    bco.createMatchList(imagesPath, mergedValidSeqsGraph, matchListFile, imageListSuffix=imageListSuffix)

    # run custom match
    run_custom_matcher(colmapPath, dbFile, matchListFile, min_num_inliers=matcher_min_num_inliers)

    # run image registrator
    inputModelPath = bco.getOneValidModelPath([os.path.join(modelPath, refName + imageListSuffix)])
    run_image_registrator(colmapPath, dbFile, inputModelPath, mergedModelPath,
                          min_num_matches=mapper_min_num_matches,
                          init_min_num_inliers=mapper_init_min_num_inliers,
                          abs_pose_min_num_inliers=mapper_abs_pose_min_num_inliers)
    
    # update_loc_model_id_refer_to_locmap_database
    localModelPath = bco.getOneValidModelPath([os.path.join(modelPath, localName + imageListSuffix)])
    time.sleep(2)
    localModelPathTmp = localModelPath + '/tmp'
    update_loc_model_id_refer_to_locmap_database(localModelPath, dbFile, localModelPathTmp)
    
    # run model merger
    localModelPath = bco.getOneValidModelPath([os.path.join(modelPath, localName + imageListSuffix)])
    mapModelPath = bco.getOneValidModelPath([mergedModelPath])

    # run_model_merger(colmapPath, localModelPathTmp, mapModelPath, mergedModelPath)
    run_model_merger_plus(colmapPath, dbFile, imagesPath, localModelPathTmp, mapModelPath, mergedModelPath, max_reproj_error=merger_max_reproj_error)
    shutil.rmtree(localModelPathTmp)
    # run colmapSFM
    # inputModelPath = bco.getOneValidModelPath([mergedModelPath])
    # run_custom_sfm(colmapPath, dbFile, imagesPath, imageListFile, matchListFile, mergedModelPath,
    #                matcher_min_num_inliers=matcher_min_num_inliers, mapper_min_num_matches=mapper_min_num_matches,
    #                mapper_init_min_num_inliers=mapper_init_min_num_inliers,
    #                mapper_abs_pose_min_num_inliers=mapper_abs_pose_min_num_inliers,
    #                mapper_input_model_path=inputModelPath,
    #                skip_feature_extractor=skip_feature_extractor,
    #                apply_model_aligner=apply_model_aligner,
    #                mask_path=mask_path)

    return

def mergedSFM(localName, refName, mergedName, mergedValidSeqsGraph, imagesPath, dbPath, modelPath, colmapPath="colmap",
              matcher_min_num_inliers=15,
              mapper_min_num_matches=45,
              mapper_init_min_num_inliers=200,
              mapper_abs_pose_min_num_inliers=60,
              skip_feature_extractor=True,
              apply_model_aligner=False,
              imageListSuffix='',
              mask_path='',
              mapper_fix_existing_images=0,
              optionalMapper=False,
              usePartialExistingModel=False,
              assocSeqs=[]):
    # merge imageList and db
    imageListFile = os.path.join(imagesPath, mergedName + imageListSuffix + '.txt')
    dbFile = os.path.join(dbPath, mergedName + imageListSuffix + '.db')
    mergedModelPath = os.path.join(modelPath, mergedName + imageListSuffix)
    if os.path.isdir(mergedModelPath):
        shutil.rmtree(mergedModelPath)
    os.mkdir(mergedModelPath)

    subNodeNamesInDict = [refName, localName]
    subListFiles = []
    subDBFiles = []
    for subNodeNameInDict in subNodeNamesInDict:
        subImageListFile = os.path.join(imagesPath, subNodeNameInDict + imageListSuffix + '.txt')
        subListFiles.append(subImageListFile)

        subDBFile = os.path.join(dbPath, subNodeNameInDict + imageListSuffix + '.db')
        subDBFiles.append(subDBFile)

    bco.mergeImageListFile(subListFiles, imageListFile)
    bco.mergeDBFile(subDBFiles, dbFile, colmapPath=colmapPath)

    # create matchList
    matchListFile = os.path.join(dbPath, mergedName + imageListSuffix + '_match.txt')
    bco.createMatchList(imagesPath, mergedValidSeqsGraph, matchListFile, imageListSuffix=imageListSuffix)

    # run colmapSFM
    inputModelPath = bco.getOneValidModelPath([os.path.join(modelPath, refName + imageListSuffix)])

    assocImagesListFile = os.path.join(imagesPath, localName + '_assoc_with_' + refName + imageListSuffix + '.txt')
    locImagesListFile = os.path.join(imagesPath, localName + imageListSuffix + '.txt')
    locAndAssocImagesListFile = os.path.join(imagesPath, localName + '_and_assoced' + imageListSuffix + '.txt')

    if usePartialExistingModel:
        subListFiles = []
        for subNodeNameInDict in assocSeqs:
            subImageListFile = os.path.join(imagesPath, subNodeNameInDict + imageListSuffix + '.txt')
            subListFiles.append(subImageListFile)
        
        bco.mergeImageListFile(subListFiles, assocImagesListFile)
        bco.mergeImageListFile([assocImagesListFile, locImagesListFile], locAndAssocImagesListFile)
        imageListFile=locAndAssocImagesListFile

    run_custom_sfm(colmapPath, dbFile, imagesPath, imageListFile, matchListFile, mergedModelPath,
                   matcher_min_num_inliers=matcher_min_num_inliers, mapper_min_num_matches=mapper_min_num_matches,
                   mapper_init_min_num_inliers=mapper_init_min_num_inliers,
                   mapper_abs_pose_min_num_inliers=mapper_abs_pose_min_num_inliers,
                   mapper_input_model_path=inputModelPath,
                   skip_feature_extractor=skip_feature_extractor,
                   apply_model_aligner=apply_model_aligner,
                   mask_path=mask_path,
                   mapper_fix_existing_images=mapper_fix_existing_images,
                   optionalMapper=optionalMapper,
                   use_partial_existing_model=usePartialExistingModel,
                   selected_existing_image_list_path=assocImagesListFile)
    return

def getAssociatedEdges(curSubNodeName, mergedSubNodeNames, edges):
    assocEdges = []
    redundantEdges = []
    for name in mergedSubNodeNames:
        redundantEdges.append([curSubNodeName, name])

    for edge in edges:
        for tarEdge in redundantEdges:
            if tt.matchedEdge(edge, tarEdge):
                assocEdges.append(edge)
                break

    return assocEdges

def getSubNodePairsWithInDictName(mapNode, subNodePairs):
    outPairs = []

    for i in range(len(subNodePairs)):
        name1 = tt.getSubNodeNameInDict(mapNode.inheritNames, mapNode.type, subNodePairs[i][0])
        name2 = tt.getSubNodeNameInDict(mapNode.inheritNames, mapNode.type, subNodePairs[i][1])

        outPairs.append([name1, name2])

    return outPairs

def getValidSeqsGraphOfRoutePairs(routePairs, mapNodes, parentNodeType='', curNodeName='', assocSeqs=[]):
    outGraph = []
    assocRoutes = []

    for routePair in routePairs:
        routeNode1 = mapNodes[routePair[0]]
        routeNode2 = mapNodes[routePair[1]]
        outGraph += bco.mergeTwoMatchGraph(routeNode1.validSeqs, routeNode2.validSeqs,
                                           [], [])# 每个route内部的validSeqsGraph无需继承到validSeqsGraphRoutePair中
        if parentNodeType=='place':
            for routeName in routePair:
                if (not (routeName == curNodeName)) and (not (routeName in assocRoutes)):
                    assocRoutes.append(routeName)
                    assocSeqs += mapNodes[routeName].validSeqs
        elif parentNodeType=='scene':
            for routeName in routePair:
                parentName = mapNodes[routeName].parent
                if (not (parentName == curNodeName))  and (not (routeName in assocRoutes)):
                    assocRoutes.append(routeName)
                    assocSeqs += mapNodes[routeName].validSeqs

    return outGraph

def saveStrList(strList, filePath):
    with open(filePath, 'w') as fp:
        for str in strList:
            fp.write(str + '\n')

    return

def readStrList(filePath):
    lines = []
    with open(filePath) as fp:
        lines = fp.readlines()

    validStrList = []
    for line in lines:
        line = line.strip()
        if (len(line) > 1) and (not (line[0]=='#')):
            validStrList.append(line)

    return validStrList

def incrementalGroupedSFM(nodeNameInDict, subNodeNamesInDict, mapNodes, imagesPath, dbPath, modelPath, removeTmpData=False,
                          colmapPath="colmap",
                          matcher_min_num_inliers=15,
                          mapper_min_num_matches=45,
                          mapper_init_min_num_inliers=200,
                          mapper_abs_pose_min_num_inliers=60,
                          skip_feature_extractor=True,
                          apply_model_aligner=False,
                          imageListSuffix='',
                          mask_path='',
                          useMergedSFMPlus=False,
                          merger_max_reproj_error=64,
                          mapper_fix_existing_images=0,
                          startMergeId=1,
                          optionalMapper=False,
                          usePartialExistingModel=False):

    mapNode = mapNodes[nodeNameInDict]

    # analyze merge order of subNodes
    mergeOrders = mapNode.mergeOrders.copy()
    assert len(mergeOrders) == len(mapNode.subNodes)
    
    subNodesToMerge = tt.getSubNodeNamesInDict(mapNode.inheritNames, mapNode.type, mergeOrders)

    # incrementally merged SFM
    lastMergedVirtualNode = subNodesToMerge[0]
    lastValidSeqsGraph = mapNodes[subNodesToMerge[0]].validSeqsGraph
    mergedSubNodeNames = []
    mergedSubNodeNames.append(mergeOrders[0])
    saveStrList(mergeOrders, os.path.join(dbPath, nodeNameInDict+'_mergeOrders.txt'))

    tmpDataNameList = []
    for i in range(1, len(mergeOrders)):
        curSubNodeName = mergeOrders[i]
        curSubNodeNameInDict = tt.getSubNodeNameInDict(mapNode.inheritNames, mapNode.type, curSubNodeName)

        if i == len(mergeOrders)-1:
            curMergedVirtualNode = nodeNameInDict
        else:
            curMergedVirtualNode = nodeNameInDict + '_tmp%d' % i
            tmpDataNameList.append(curMergedVirtualNode)

        # get merged validSeqsGraph
        assocEdges = getAssociatedEdges(curSubNodeName, mergedSubNodeNames, mapNode.edges)

        if mapNode.type == 'place':
            assocRoutePairs = getSubNodePairsWithInDictName(mapNode, assocEdges)
        elif mapNode.type == 'scene':
            assocRoutePairs = tt.getSceneHooksByEdge(mapNode.name, assocEdges, mapNodes)
        else:
            raise Exception('unsupported node type for now.')
        
        assocSeqs = []
        assocValidSeqsGraph = getValidSeqsGraphOfRoutePairs(assocRoutePairs, mapNodes,
                                                            parentNodeType=mapNode.type,
                                                            curNodeName=curSubNodeNameInDict,
                                                            assocSeqs=assocSeqs)
        # run mergeSFM
        if (i >= startMergeId):
            if useMergedSFMPlus:
                mergedSFMPlus(curSubNodeNameInDict, lastMergedVirtualNode, curMergedVirtualNode, assocValidSeqsGraph,
                          imagesPath, dbPath, modelPath,
                          colmapPath=colmapPath,
                          matcher_min_num_inliers=matcher_min_num_inliers,
                          mapper_min_num_matches=mapper_min_num_matches,
                          mapper_init_min_num_inliers=mapper_init_min_num_inliers,
                          mapper_abs_pose_min_num_inliers=mapper_abs_pose_min_num_inliers,
                          skip_feature_extractor=skip_feature_extractor,
                          apply_model_aligner=apply_model_aligner,
                          imageListSuffix=imageListSuffix,
                          mask_path=mask_path,
                          merger_max_reproj_error=merger_max_reproj_error)
            else:
                mergedSFM(curSubNodeNameInDict, lastMergedVirtualNode, curMergedVirtualNode, assocValidSeqsGraph,
                          imagesPath, dbPath, modelPath,
                          colmapPath=colmapPath,
                          matcher_min_num_inliers=matcher_min_num_inliers,
                          mapper_min_num_matches=mapper_min_num_matches,
                          mapper_init_min_num_inliers=mapper_init_min_num_inliers,
                          mapper_abs_pose_min_num_inliers=mapper_abs_pose_min_num_inliers,
                          skip_feature_extractor=skip_feature_extractor,
                          apply_model_aligner=apply_model_aligner,
                          imageListSuffix=imageListSuffix,
                          mask_path=mask_path,
                          mapper_fix_existing_images=mapper_fix_existing_images,
                          optionalMapper=optionalMapper,
                          usePartialExistingModel=usePartialExistingModel,
                          assocSeqs=assocSeqs)

        # prepare for next merge
        lastMergedVirtualNode = curMergedVirtualNode
        lastValidSeqsGraph = lastValidSeqsGraph + assocValidSeqsGraph
        mergedSubNodeNames.append(curSubNodeName)
        saveStrList(mergedSubNodeNames, os.path.join(dbPath, curMergedVirtualNode + '_mergeOrders.txt'))

    # update validSeqsGraph for nodeNameInDict
    mapNodes[nodeNameInDict].validSeqsGraph = lastValidSeqsGraph.copy()

    # clean tmp data
    if removeTmpData:
        bco.cleanColmapMaterialsByNames(tmpDataNameList, imagesPath, dbPath, modelPath, imageListSuffix=imageListSuffix)

    return

def groupedSFM(nodeNameInDict, subNodeNamesInDict, mapNodes, imagesPath, dbPath, modelPath,
               colmapPath="colmap",
               matcher_min_num_inliers=15,
               mapper_min_num_matches=45,
               mapper_init_min_num_inliers=200,
               mapper_abs_pose_min_num_inliers=60,
               skip_feature_extractor=True,
               apply_model_aligner=False,
               imageListSuffix='',
               mask_path='',
               useMergedSFMPlus=False,
               merger_max_reproj_error=64,
               mapper_fix_existing_images=0,
               startMergeId=1,
               optionalMapper=False,
               useSim3GroupedScene=False,
               sim3PoseGraphApp='',
               usePartialExistingModel=False):
    mapNode = mapNodes[nodeNameInDict]

    if mapNode.type == 'route':
        globalGroupedSFM(nodeNameInDict, subNodeNamesInDict, mapNodes, imagesPath, dbPath, modelPath,
                         colmapPath=colmapPath,
                         matcher_min_num_inliers=matcher_min_num_inliers,
                         mapper_min_num_matches=mapper_min_num_matches,
                         mapper_init_min_num_inliers=mapper_init_min_num_inliers,
                         mapper_abs_pose_min_num_inliers=mapper_abs_pose_min_num_inliers,
                         skip_feature_extractor=skip_feature_extractor,
                         apply_model_aligner=apply_model_aligner,
                         imageListSuffix=imageListSuffix,
                         mask_path=mask_path,
                         optionalMapper=optionalMapper)
    else:
        if (mapNode.type == 'scene') and useSim3GroupedScene:
            sim3GroupedSFM(nodeNameInDict, subNodeNamesInDict, mapNodes, imagesPath, dbPath, modelPath,
                           colmapPath=colmapPath,
                           matcher_min_num_inliers=matcher_min_num_inliers,
                           mapper_min_num_matches=mapper_min_num_matches,
                           mapper_init_min_num_inliers=mapper_init_min_num_inliers,
                           mapper_abs_pose_min_num_inliers=mapper_abs_pose_min_num_inliers,
                           mapper_fix_existing_images=mapper_fix_existing_images,
                           imageListSuffix=imageListSuffix,
                           mask_path=mask_path,
                           sim3PoseGraphApp=sim3PoseGraphApp)
        else:
            incrementalGroupedSFM(nodeNameInDict, subNodeNamesInDict, mapNodes, imagesPath, dbPath, modelPath,
                                colmapPath=colmapPath,
                                matcher_min_num_inliers=matcher_min_num_inliers,
                                mapper_min_num_matches=mapper_min_num_matches,
                                mapper_init_min_num_inliers=mapper_init_min_num_inliers,
                                mapper_abs_pose_min_num_inliers=mapper_abs_pose_min_num_inliers,
                                skip_feature_extractor=skip_feature_extractor,
                                apply_model_aligner=apply_model_aligner,
                                imageListSuffix=imageListSuffix,
                                mask_path=mask_path,
                                useMergedSFMPlus=useMergedSFMPlus,
                                merger_max_reproj_error=merger_max_reproj_error,
                                mapper_fix_existing_images=mapper_fix_existing_images,
                                startMergeId=startMergeId,
                                optionalMapper=optionalMapper,
                                usePartialExistingModel=usePartialExistingModel)

    return

def node_scale_reconstruction(tmpDataPath, imagesPath, modelPath, boardParam, charucoSeqs,
                              colmap_exe='colmap', imageListSuffix=''):
    if not os.path.exists(tmpDataPath):
        os.makedirs(tmpDataPath)

    curNodeModelPath = bco.getOneValidModelPath([modelPath])
    if os.path.exists(curNodeModelPath) and (len(charucoSeqs) > 0):
        scale = get_charuco_scale(tmpDataPath, imagesPath, curNodeModelPath, boardParam, charucoSeqs,
                                  colmap_exe=colmap_exe, image_list_suffix=imageListSuffix)
        if scale > 0:
            rescale_model(scale, curNodeModelPath, curNodeModelPath)
        else:
            print("warning! minus scale returned: scale = " + str(scale) + '\n')

    return

def node_orientation_aligner(colmapPath, imagesPath, modelPath, maxImageSize=1024):
    curNodeModelPath = bco.getOneValidModelPath([modelPath])

    if os.path.exists(curNodeModelPath):
        bco.run_model_orientation_aligner(colmapPath, imagesPath, curNodeModelPath, curNodeModelPath,
                                          max_image_size=maxImageSize,
                                          rList=[1, 0, 0, 0, 0, 1, 0, -1, 0], tList=[0, 0, 0])

    return

def node_model_deleter(imagesPath, modelPath, nodeName, charucoSeqs, colmapPath='colmap', imageListSuffix='', outModelSuffix='_deleted'):
    inputModelPathLeaf = bco.getOneValidModelPath([os.path.join(modelPath, nodeName+imageListSuffix)])
    outModelPath = os.path.join(modelPath, nodeName+outModelSuffix+imageListSuffix)
    outModelPathLeaf = os.path.join(outModelPath, '0')

    if os.path.isdir(outModelPath):
        shutil.rmtree(outModelPath)
    os.makedirs(outModelPathLeaf)

    subListFiles = []
    for seq in charucoSeqs:
        subListFiles.append(os.path.join(imagesPath, seq+imageListSuffix+'.txt'))
    
    imagesToDelListFile = os.path.join(imagesPath, nodeName+'_todel'+imageListSuffix+'.txt')
    bco.mergeImageListFile(subListFiles, imagesToDelListFile)

    run_image_deleter(colmapPath, inputModelPathLeaf, outModelPathLeaf, '', imagesToDelListFile)

    return

def node_base_reconstruction(nodeInfo, mapNodes, imagesPath, dbPath, modelPath, taskPath, boardParam,
                             colmapPath="colmap",
                             matcher_min_num_inliers=50,
                             mapper_min_num_matches=45,
                             mapper_init_min_num_inliers=200,
                             mapper_abs_pose_min_num_inliers=60,
                             skip_feature_extractor=True,
                             apply_model_aligner=False,
                             imageListSuffix='',
                             mask_path='',
                             useMergedSFMPlus=False,
                             merger_max_reproj_error=64,
                             mapper_fix_existing_images=0,
                             optionalMapper=False,
                             useSim3GroupedScene=False,
                             useExhaustiveSFM=False,
                             sim3PoseGraphApp='',
                             usePartialExistingModel=False,
                             deleteCharuco=False):
    # get node in dict
    nodeNameInDict = tt.getNodeNameInDict(nodeInfo['inheritNames'], nodeInfo['type'])
    mapNode = mapNodes[nodeNameInDict]

    timeStartOriBaseRecon = time.time()
    print('start node base reconstruction of %s at time: %s\n' % (nodeNameInDict, str(timeStartOriBaseRecon)))

    if mapNode.runBaseRecon:
        # original base recon
        if useExhaustiveSFM:
            globalExhaustiveSFM(nodeNameInDict, mapNodes, imagesPath, dbPath, modelPath,
                        colmapPath=colmapPath,
                        matcher_min_num_inliers=matcher_min_num_inliers,
                        mapper_min_num_matches=mapper_min_num_matches,
                        mapper_init_min_num_inliers=mapper_init_min_num_inliers,
                        mapper_abs_pose_min_num_inliers=mapper_abs_pose_min_num_inliers,
                        skip_feature_extractor=False,
                        apply_model_aligner=apply_model_aligner,
                        imageListSuffix=imageListSuffix,
                        mask_path=mask_path)
        else:
            if len(mapNode.subNodes) == 0:
                raise Exception("len(mapNode.subNodes)==0, unsupported node for now.\n")
            elif len(mapNode.subNodes) == 1:
                # copy sub node db&model as current node db&model
                subNodeNameInDict = tt.getSubNodeNameInDict(nodeInfo['inheritNames'], nodeInfo['type'], mapNode.subNodes[0])
                bco.copyAllColmapMaterials(subNodeNameInDict, nodeNameInDict, imagesPath, dbPath, modelPath,
                                        imageListSuffix=imageListSuffix,
                                        optionalMapper=(optionalMapper and (not mapNode.shouldRunMapper)))
            else:
                # groupedSFM
                subNodeNamesInDict = tt.getSubNodeNamesInDict(nodeInfo['inheritNames'], nodeInfo['type'], mapNode.subNodes)

                if nodeInfo['type'] == 'scene':
                    apply_model_aligner = True

                startMergeId = 1
                if 'startMergeId' in nodeInfo.keys():
                    startMergeId = nodeInfo['startMergeId']
                    assert startMergeId >= 1, 'startMergeId is supposed to be >= 1'

                groupedSFM(nodeNameInDict, subNodeNamesInDict, mapNodes, imagesPath, dbPath, modelPath,
                        colmapPath=colmapPath,
                        matcher_min_num_inliers=matcher_min_num_inliers,
                        mapper_min_num_matches=mapper_min_num_matches,
                        mapper_init_min_num_inliers=mapper_init_min_num_inliers,
                        mapper_abs_pose_min_num_inliers=mapper_abs_pose_min_num_inliers,
                        skip_feature_extractor=skip_feature_extractor,
                        apply_model_aligner=apply_model_aligner,
                        imageListSuffix=imageListSuffix,
                        mask_path=mask_path,
                        useMergedSFMPlus=useMergedSFMPlus,
                        merger_max_reproj_error=merger_max_reproj_error,
                        mapper_fix_existing_images=mapper_fix_existing_images,
                        startMergeId=startMergeId,
                        optionalMapper=(optionalMapper and (not mapNode.shouldRunMapper)),
                        useSim3GroupedScene=useSim3GroupedScene,
                        sim3PoseGraphApp=sim3PoseGraphApp,
                        usePartialExistingModel=usePartialExistingModel)

    print('time cost for original base reconstruction of %s is %s s\n' % (nodeNameInDict, str(time.time() - timeStartOriBaseRecon)))

    # scale reconstruction for place node
    shouldHaveModel = (not optionalMapper) or mapNode.shouldRunMapper or useExhaustiveSFM
    shouldRunScaleAndGravityRec = (((nodeInfo['type'] == 'place') or ((nodeInfo['type'] == 'scene') and useExhaustiveSFM)) and shouldHaveModel)

    nodeModelPath = os.path.join(modelPath, nodeNameInDict + imageListSuffix)
    if (shouldRunScaleAndGravityRec and (not (mapNode.runScaleRec == 'negative'))) or (mapNode.runScaleRec == 'positive'):
        # charuco scale recon
        tmpDataPath = os.path.join(taskPath, nodeNameInDict)
        if not os.path.exists(tmpDataPath):
            os.makedirs(tmpDataPath)

        timeStartScaleRec = time.time()
        print('start node scale reconstruction of %s at time: %s\n' % (nodeNameInDict, str(timeStartScaleRec)))

        node_scale_reconstruction(tmpDataPath, imagesPath, nodeModelPath, boardParam, mapNode.charucoSeqs,
                                colmap_exe=colmapPath, imageListSuffix=imageListSuffix)

        print('time cost for scale reconstruction of %s is %s s\n' % (nodeNameInDict, str(time.time() - timeStartScaleRec)))

    if (shouldRunScaleAndGravityRec and (not (mapNode.runGravityRec == 'negative'))) or (mapNode.runGravityRec == 'positive'):
        # gravity alignment
        timeStartGravityRec = time.time()
        print('start node gravity reconstruction of %s at time: %s\n' % (nodeNameInDict, str(timeStartGravityRec)))
        node_orientation_aligner(colmapPath, imagesPath, nodeModelPath, maxImageSize=1024)
        print('time cost for gravity reconstruction of %s is %s s\n' % (nodeNameInDict, str(time.time() - timeStartGravityRec)))

    # delete charuco sequences in sparse model of scene node
    shouldDeleteCharuco = (nodeInfo['type'] == 'scene') and shouldHaveModel and deleteCharuco
    if (shouldDeleteCharuco and (not (mapNode.runCharucoDelete == 'negative'))) or (mapNode.runCharucoDelete == 'positive'):
        node_model_deleter(imagesPath, modelPath, nodeNameInDict, mapNode.charucoSeqs, 
                           colmapPath=colmapPath, imageListSuffix=imageListSuffix, outModelSuffix='_CharucoDeleted')

    print('time cost for whole reconstruction of %s is %s s\n' % (nodeNameInDict, str(time.time() - timeStartOriBaseRecon)))
    return

def run_base_reconstruction(args):
    # logger
    logStdoutFile = os.path.join(args.projPath, 'logStdout_base_reconstruction.txt')
    sys.stdout = Logstdout(logStdoutFile, sys.stdout)

    print("start--->run_base_reconstruction\n")

    timeVeryBeginning = time.time()

    imagesPath = os.path.join(args.projPath, args.imagesDir)
    dbPath = os.path.join(args.projPath, args.dbDir)
    modelPath = os.path.join(args.projPath, args.modelDir)
    taskPath = os.path.join(args.projPath, args.taskDir, 'baseRecon')
    masksPath = ''
    if not (args.masksDir==None):
        masksPath = os.path.join(args.projPath, args.masksDir)

    # if there exists suppleGraphFile, running with map supplement mode
    mapSuppleMode = False
    suppleGraphFile = os.path.join(args.projPath, args.supplementalGraph+'.json')
    if os.path.isfile(suppleGraphFile):
        mapSuppleMode = True

    # parse scene graph
    taskPath = os.path.join(args.projPath, args.taskDir, 'baseRecon')
    configPath = os.path.join(args.projPath, args.configDir)
    supplePlaces = []
    if args.validSeqsFromJson:
        sceneTopology = tt.getSceneTopology(args.projPath, configPath=configPath, sceneGraphName=args.sceneGraph,
                                            validSeqsFromJson=args.validSeqsFromJson,
                                            suppleGraphName=args.supplementalGraph,
                                            supplePlaces=supplePlaces,
                                            imagesDir=args.imagesDir,
                                            taskPath=taskPath)
    else:
        sceneTopology = tt.getSceneTopology(args.projPath, projName=args.projName, batchName=args.batchName, sceneGraphName=args.sceneGraph,
                                            suppleGraphName=args.supplementalGraph,
                                            supplePlaces=supplePlaces,
                                            imagesDir=args.imagesDir,
                                            taskPath=taskPath)

    tt.saveImageListForMatch(imagesPath, sceneTopology[3], imageListSuffix=args.imageListSuffix)
    mapNodes = tt.getSerializedMapNodes(sceneTopology)

    # get base reconstruction tasks
    if not (args.baseReconConfig==None):
        if os.path.exists(args.baseReconConfig):
            seqToReconList, routeToReconList, placeToReconList, sceneToReconList = \
                tt.getBaseReconTasksSpecified(args.baseReconConfig, sceneTopology, dbPath)
        else:
            raise Exception('Specified baseReconConfig does not exist: %s\n' % args.baseReconConfig)
    else:
        if mapSuppleMode:
            seqToReconList, routeToReconList, placeToReconList, sceneToReconList = tt.getBaseReconTasksMapSupple(sceneTopology, supplePlaces)
        else:
            seqToReconList, routeToReconList, placeToReconList, sceneToReconList = tt.getBaseReconTasks(sceneTopology)
            if args.useExhaustiveSFM:
                seqToReconList = []
                routeToReconList = []
                placeToReconList = []

    # update mergeOrders of scene node if mapSuppleMode=True
    # tips1: the mergeOrders may be already specified by reconlist
    # tips2: the edges may be not interconnected for supplemental places
    # tips3: the existing colmap data of scene node should be copied as colmap data of sceneNode_tmp%d
    # tips4: the finally updated colmap data must not cover existing colmap data of scene node
    # tips5: save the supplemented scene_graph
    sceneNodeNameBkp = ''
    if len(sceneToReconList)>0:
        sceneNodeNameBkp = sceneToReconList[0]['inheritNames'][0] + '_bkp_' + time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))

    if mapSuppleMode:
        mergeOrdersUpdated = tt.updateSceneNodeMergeOrders(supplePlaces, sceneToReconList[0], mapNodes)
        tt.setChildNodesShouldRunMapper(sceneTopology)

        if sceneToReconList[0]['startMergeId']>1:
            lastMergedTmpNodeName = sceneToReconList[0]['inheritNames'][0] + '_tmp%d' % (sceneToReconList[0]['startMergeId']-1)
        else:
            lastMergedTmpNodeName = mergeOrdersUpdated[0]

        lastMergedTmpDBFile = os.path.join(dbPath, lastMergedTmpNodeName+args.imageListSuffix+'.db')
        
        sceneNode = mapNodes[sceneToReconList[0]['inheritNames'][0]]
        if (sceneToReconList[0]['startMergeId'] == (len(sceneNode.subNodes)-len(supplePlaces))) and \
           (not os.path.isfile(lastMergedTmpDBFile) and \
            not  args.useSim3GroupedScene):
            bco.copyAllColmapMaterials(sceneToReconList[0]['inheritNames'][0] + args.inMapSuffix, lastMergedTmpNodeName, imagesPath, dbPath, modelPath,
                                       imageListSuffix=args.imageListSuffix)
        
        # rename the existing colmap data
        bco.renameAllColmapMaterials(sceneToReconList[0]['inheritNames'][0], sceneNodeNameBkp, imagesPath, dbPath, modelPath,
                                      imageListSuffix=args.imageListSuffix)
        
        # save supplemented scene_graph
        mergedSceneGraphFile = os.path.join(args.projPath, args.sceneGraph+'_'+args.supplementalGraph+'.json')
        tt.writeSceneTolopogy(sceneTopology, mergedSceneGraphFile)

    # update shouldRunMapper flags of place and associated routes and seqs
    if args.useSim3GroupedScene:
        tt.setPlaceNodesShouldRunMapper(sceneTopology, shouldRunMapper=True)

    # check existentce of sequence imageDir
    bco.checkExistence(seqToReconList, imagesPath)

    # sequence reconstruction
    timeStartSeqRecon = time.time()
    print('start sequences base reconstruction at: ' + str(timeStartSeqRecon) + '\n')

    seqRunMapperFlags = tt.getRunMapperFlags(seqToReconList, tt.getSerializedMapNodes(sceneTopology))
    run_colmap_seq_sfms(args.colmapPath, imagesPath, dbPath, modelPath, seqToReconList,
                      seq_run_mapper_flags=seqRunMapperFlags, suffix_str=args.imageListSuffix,
                      seq_match_overlap=20, mapper_min_num_matches=50,
                      mapper_init_min_num_inliers=200, mapper_abs_pose_min_num_inliers=60,
                      mask_path=masksPath, optionalMapper=args.optionalMapper)
    print('time cost for sequences base reconstruction = ' + str(time.time() - timeStartSeqRecon) + 's\n')

    # route, place and scene reconstruction
    # addition order is important!!!!
    mergedNodeList = routeToReconList + placeToReconList + sceneToReconList

    for nodeInfo in mergedNodeList:
        node_base_reconstruction(nodeInfo, mapNodes, imagesPath, dbPath, modelPath, taskPath, args.boardParam,
                                 colmapPath=args.colmapPath,
                                 mapper_min_num_matches=50, mapper_init_min_num_inliers=200,
                                 imageListSuffix=args.imageListSuffix,
                                 mask_path=masksPath,
                                 useMergedSFMPlus=args.useMergedSFMPlus,
                                 merger_max_reproj_error=64,
                                 mapper_fix_existing_images=1,
                                 optionalMapper=args.optionalMapper,
                                 useSim3GroupedScene=args.useSim3GroupedScene,
                                 useExhaustiveSFM=args.useExhaustiveSFM,
                                 sim3PoseGraphApp=args.sim3PoseGraphApp,
                                 usePartialExistingModel=args.usePartialExistingModel,
                                 deleteCharuco=(not mapSuppleMode))

    if mapSuppleMode and (not (args.outMapSuffix==None)):
        sceneNodeNameUpdate = sceneToReconList[0]['inheritNames'][0] + args.outMapSuffix
        bco.renameAllColmapMaterials(sceneToReconList[0]['inheritNames'][0], sceneNodeNameUpdate, imagesPath, dbPath, modelPath,
                                      imageListSuffix=args.imageListSuffix)
        
        bco.renameAllColmapMaterials(sceneNodeNameBkp, sceneToReconList[0]['inheritNames'][0], imagesPath, dbPath, modelPath,
                                      imageListSuffix=args.imageListSuffix)

    totalTimeCost = time.time() - timeVeryBeginning
    print("<base_reconstruction>totalTimeCost = " + str(totalTimeCost) + "s\n")
    return

def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--projPath', type=str, required=True)
    parser.add_argument('--projName', type=str, default=None, help='如果使用了--validSeqsFromJson，则必须指定该参数')
    parser.add_argument('--batchName', type=str, default=None, help='如果使用了--validSeqsFromJson，则必须指定该参数')
    parser.add_argument('--boardParam', type=str, required=True)
    parser.add_argument('--sceneGraph', type=str, default='scene_graph_checked')
    parser.add_argument('--supplementalGraph', type=str, default='')
    parser.add_argument('--outMapSuffix', default=None, type=str)
    parser.add_argument('--inMapSuffix', default='', type=str)
    parser.add_argument('--usePartialExistingModel', action="store_true")

    parser.add_argument('--imagesDir', default='images')
    parser.add_argument('--masksDir', default=None, type=str)
    parser.add_argument('--dbDir', default='database')
    parser.add_argument('--modelDir', default='sparse')
    parser.add_argument('--taskDir', default='tasks')
    parser.add_argument('--imageListSuffix', default='', type=str)
    parser.add_argument('--configDir', default='config')
    parser.add_argument('--validSeqsFromJson', action='store_false')

    parser.add_argument('--colmapPath', default="colmap")
    parser.add_argument('--baseReconConfig', default=None, type=str)
    parser.add_argument('--useMergedSFMPlus', action='store_true')
    parser.add_argument('--useSim3GroupedScene', action='store_true')
    parser.add_argument('--optionalMapper', action='store_false')
    parser.add_argument('--useExhaustiveSFM', action='store_true')
    parser.add_argument('--sim3PoseGraphApp', default='',type=str)

    args = parser.parse_args()

    return args

def main():
    args = parseArgs()
    run_base_reconstruction(args)

if __name__ == '__main__':
    main()