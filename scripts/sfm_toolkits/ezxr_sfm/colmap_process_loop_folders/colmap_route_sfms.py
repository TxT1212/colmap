# coding: utf-8
import os,sys
import shutil
sys.path.append('../')
from colmap_process.colmap_seq_sfm import run_route_sfm, run_mapper, run_image_registrator, run_model_merger, run_model_sim3_merger, run_custom_matcher, run_bundle_adjuster, run_point_triangulator
from colmap_process.colmap_model_modify import update_loc_model_id_refer_to_locmap_database
from colmap_process_loop_folders.basic_colmap_operation import getOneValidModelPath

def run_database_merger(colmap_exe, loc_db, map_db, locmap_db):
    '''
    database合并, 调用colmap的命令行
    '''
    if (locmap_db == loc_db) or (locmap_db == map_db):
        raise Exception('unsupported case: locmap_db==loc_db? or locmap_db==map_db?')
    else:
        if os.path.isfile(locmap_db):
            os.remove(locmap_db)

    run_str = colmap_exe + ' database_merger --database_path1 ' + map_db + \
        ' --database_path2 ' + loc_db + \
        ' --merged_database_path ' + locmap_db
    print(run_str)
    os.system(run_str)
    return

def readImageList(listFile):
    imageList = []
    with open(listFile) as fp:
        imageList = fp.readlines()

    for i in range(len(imageList)):
        imageList[i] = imageList[i].strip()

    return imageList

def writeStrList(imageList, outFile):
    with open(outFile, 'w') as fp:
        for image in imageList:
            fp.write(image + '\n')

def mergeImageListFile(subListFiles, outFile):
    imageList = []

    for listFile in subListFiles:
        imageList = imageList + readImageList(listFile)

    writeStrList(imageList, outFile)

def mergeDBFile(subDBFiles, outDBFile, colmapPath='colmap', removeTmpDB=True):
    outDBDir, outDBFileName = os.path.split(outDBFile)
    outDBName, _ = os.path.splitext(outDBFileName)

    if len(subDBFiles) == 1:
        shutil.copy(subDBFiles[0], outDBFile)
    elif len(subDBFiles)>1:
        tmpDBFiles = []
        lastMergedDBFile = subDBFiles[0]
        for i in range(1, len(subDBFiles)):
            tmpDBFile = os.path.join(outDBDir, outDBName+'_%d.db' % (i))
            tmpDBFiles.append(tmpDBFile)

            run_database_merger(colmapPath, subDBFiles[i], lastMergedDBFile, tmpDBFile)

            lastMergedDBFile = tmpDBFile

        if not (lastMergedDBFile == outDBFile):
            shutil.copy(lastMergedDBFile, outDBFile)

        if removeTmpDB:
            for tmpDBFile in tmpDBFiles:
                os.remove(tmpDBFile)

    print('%d database are merged to %s\n' % (len(subDBFiles), outDBFile))

def mergeTwoMatchGraph(nodes1, nodes2, graph1, gprah2):
    outGraph = graph1 + gprah2
    for node1 in nodes1:
        for node2 in nodes2:
            outGraph.append([node1, node2])

    return outGraph

def matchTwoImageList(imageLsit1, imageList2):
    matchedList = []
    for image1 in imageLsit1:
        for image2 in imageList2:
            matchedList.append(image1 + ' ' + image2)

    return matchedList

def createMatchList(imagePath, validSeqsGraph, matchListFile, imageListSuffix=''):
    matchList = []
    for edge in validSeqsGraph:
        imageList1 = readImageList(os.path.join(imagePath, edge[0]+imageListSuffix+'.txt'))
        imageList2 = readImageList(os.path.join(imagePath, edge[1]+imageListSuffix+'.txt'))

        matchList = matchList + matchTwoImageList(imageList1, imageList2)

    writeStrList(matchList, matchListFile)

def copyAllFiles(src,dst):
    srcFiles = os.listdir(src)

    for fileName in srcFiles:
        fullFile = os.path.join(src, fileName)
        if os.path.isfile(fullFile):
            shutil.copy(fullFile, dst)

    return

def copyModelFiles(src, dst):
    srcFiles = os.listdir(src)

    for fileName in srcFiles:
        fullFile = os.path.join(src, fileName)
        if os.path.isfile(fullFile):
            shutil.copy(fullFile, dst)
        elif os.path.isdir(fullFile):
            dstSubDir = os.path.join(dst, fileName)
            if not os.path.exists(dstSubDir):
                os.mkdir(dstSubDir)
            copyAllFiles(fullFile, dstSubDir)

    return
    
def runRouteSFM(colmapPath, imagePath, dbPath, modelPath, routeInfo, \
                matcher_min_num_inliers=50, mapper_min_num_matches=45, \
                mapper_init_min_num_inliers=200, mapper_abs_pose_min_num_inliers=60, skip_feature_extractor=True,
                imageListSuffix=''):
    inheritNames = routeInfo['inheritNames']
    validSeqs = routeInfo['validSeqs']
    validSeqsGraph = routeInfo['validSeqsGraph']

    # route name in dict
    routeNameInDict = inheritNames[-2] + '_' + inheritNames[-1]

    # merge image lists and databses of seqs
    imageListFile = os.path.join(imagePath, routeNameInDict + imageListSuffix + '.txt')
    dbFile = os.path.join(dbPath, routeNameInDict + imageListSuffix + '.db')

    subListFiles = []
    subDBFiles = []
    subModelPaths = []
    for seq in validSeqs:
        seqImageListFile = os.path.join(imagePath, seq + imageListSuffix + '.txt')
        subListFiles.append(seqImageListFile)

        seqDBFile = os.path.join(dbPath, seq + imageListSuffix + '.db')
        subDBFiles.append(seqDBFile)

        subModelPath = os.path.join(modelPath, seq + imageListSuffix)
        subModelPaths.append(subModelPath)

    mergeImageListFile(subListFiles, imageListFile)
    mergeDBFile(subDBFiles, dbFile, colmapPath=colmapPath)

    # create match list for route
    matchListFile = os.path.join(dbPath, routeNameInDict + imageListSuffix + '_match.txt')
    createMatchList(imagePath, validSeqsGraph, matchListFile, imageListSuffix=imageListSuffix)

    # run colmap reconstruction
    # if there is only one valid seq for current route, copy the seq model as route model
    routeModelPath = os.path.join(modelPath, routeNameInDict + imageListSuffix)

    if os.path.exists(routeModelPath):
        shutil.rmtree(routeModelPath)
    os.mkdir(routeModelPath)

    if len(validSeqs) == 1:
        seqModelPath = os.path.join(modelPath, validSeqs[0] + imageListSuffix)
        copyModelFiles(seqModelPath, routeModelPath)
    else:
        run_route_sfm(colmapPath, dbFile, imagePath, imageListFile, matchListFile, routeModelPath,
                      matcher_min_num_inliers=matcher_min_num_inliers, mapper_min_num_matches=mapper_min_num_matches,
                      mapper_init_min_num_inliers=mapper_init_min_num_inliers,
                      mapper_abs_pose_min_num_inliers=mapper_abs_pose_min_num_inliers,
                      mapper_input_model_path='',
                      skip_feature_extractor=skip_feature_extractor)

    return

def runRoutePairSFMPlus(colmapPath, imagePath, dbPath, modelPath, routePairInfo,
                    matcher_min_num_inliers=50, mapper_min_num_matches=45,
                    mapper_init_min_num_inliers=200, mapper_abs_pose_min_num_inliers=60, skip_feature_extractor=True,
                    imageListSuffix='', merger_max_reproj_error=64, merger_min_2d_inlier_percent=0.3):
    routeNameInDictList = []
    validSeqsList = []
    validSeqsGraphList = []

    subListFiles = []
    subDBFiles = []
    subModelPaths = []

    for routeInfo in routePairInfo:
        inheritNames = routeInfo['inheritNames']
        routeNameInDict = inheritNames[-2] + '_' + inheritNames[-1]
        routeNameInDictList.append(routeNameInDict)

        validSeqs = routeInfo['validSeqs']
        validSeqsList.append(validSeqs)

        validSeqsGraph = routeInfo['validSeqsGraph']
        validSeqsGraphList.append(validSeqsGraph)

        routeImageListFile = os.path.join(imagePath, routeNameInDict + imageListSuffix + '.txt')
        subListFiles.append(routeImageListFile)

        routeDBFile = os.path.join(dbPath, routeNameInDict + imageListSuffix + '.db')
        subDBFiles.append(routeDBFile)

        routeModelPath = os.path.join(modelPath, routeNameInDict + imageListSuffix)
        subModelPaths.append(routeModelPath)

    routePairName = routeNameInDictList[0]
    for i in range(1, len(routeNameInDictList)):
        routePairName = routePairName + '_' + routeNameInDictList[i]

    # merge image lists and databses of seqs
    imageListFile = os.path.join(imagePath, routePairName + imageListSuffix + '.txt')
    dbFile = os.path.join(dbPath, routePairName + imageListSuffix + '.db')
    mergeImageListFile(subListFiles, imageListFile)
    mergeDBFile(subDBFiles, dbFile, colmapPath=colmapPath)

    # create match list for routePair
    matchListFile = os.path.join(dbPath, routePairName + imageListSuffix + '_match.txt')
    validSeqsGraphRoutePair = mergeTwoMatchGraph(validSeqsList[0], validSeqsList[1],
                                                 [], [])# 每个route内部的validSeqsGraph无需继承到validSeqsGraphRoutePair中

    createMatchList(imagePath, validSeqsGraphRoutePair, matchListFile, imageListSuffix=imageListSuffix)

    # run custom match
    run_custom_matcher(colmapPath, dbFile, matchListFile, min_num_inliers=matcher_min_num_inliers)

    # run image registrator
    routePairModelPath = os.path.join(modelPath, routePairName + imageListSuffix)
    if os.path.exists(routePairModelPath):
        shutil.rmtree(routePairModelPath)
    os.mkdir(routePairModelPath)

    inputModelPath = getOneValidModelPath([subModelPaths[0]])
    # # 用mapper的方式实现更高精度的registrator
    # run_mapper(colmapPath, dbFile, imagePath, imageListFile, routePairModelPath, \
    #     min_num_matches = 45, init_min_num_inliers = 200, abs_pose_min_num_inliers = 60, mapper_input_model_path=inputModelPath,
    #     mapper_ba_global_images_ratio=3.0, mapper_ba_global_points_ratio=3.0, mapper_fix_existing_images=1)
    run_image_registrator(colmapPath, dbFile, inputModelPath, routePairModelPath,
                          min_num_matches=mapper_min_num_matches,
                          init_min_num_inliers=mapper_init_min_num_inliers,
                          abs_pose_min_num_inliers=mapper_abs_pose_min_num_inliers)

    # update_loc_model_id_refer_to_locmap_database
    localModelPath = getOneValidModelPath([subModelPaths[1]])
    localModelPathTmp = localModelPath + '/tmp'
    update_loc_model_id_refer_to_locmap_database(localModelPath, dbFile, localModelPathTmp)

    # run model merger
    localModelPath = getOneValidModelPath([subModelPaths[1]])
    mapModelPath = getOneValidModelPath([routePairModelPath])
    run_model_merger(colmapPath, localModelPathTmp, mapModelPath, routePairModelPath, max_reproj_error=merger_max_reproj_error, min_2d_inlier_percent=merger_min_2d_inlier_percent)
    run_model_sim3_merger(colmapPath, localModelPathTmp, inputModelPath, routePairModelPath, routePairModelPath)
    shutil.rmtree(localModelPathTmp)
    # mapModelPath = getOneValidModelPath([routePairModelPath])
    # run_bundle_adjuster(colmapPath, mapModelPath, mapModelPath)
    # run_point_triangulator(colmapPath, dbFile, imagePath, mapModelPath, mapModelPath)
    # run_bundle_adjuster(colmapPath, mapModelPath, mapModelPath)

    #routePair SFM
    # inputModelPath = getOneValidModelPath([routePairModelPath])
    # run_route_sfm(colmapPath, dbFile, imagePath, imageListFile, matchListFile, routePairModelPath,
    #               matcher_min_num_inliers=matcher_min_num_inliers, mapper_min_num_matches=mapper_min_num_matches,
    #               mapper_init_min_num_inliers=mapper_init_min_num_inliers,
    #               mapper_abs_pose_min_num_inliers=mapper_abs_pose_min_num_inliers,
    #               mapper_input_model_path=inputModelPath,
    #               skip_feature_extractor=skip_feature_extractor)

    return

def runRoutePairSFM(colmapPath, imagePath, dbPath, modelPath, routePairInfo,
                    matcher_min_num_inliers=50, mapper_min_num_matches=45,
                    mapper_init_min_num_inliers=200, mapper_abs_pose_min_num_inliers=60, skip_feature_extractor=True,
                    imageListSuffix='',
                    mapper_fix_existing_images=0):
    routeNameInDictList = []
    validSeqsList = []
    validSeqsGraphList = []

    subListFiles = []
    subDBFiles = []
    subModelPaths = []

    for routeInfo in routePairInfo:
        inheritNames = routeInfo['inheritNames']
        routeNameInDict = inheritNames[-2] + '_' + inheritNames[-1]
        routeNameInDictList.append(routeNameInDict)

        validSeqs = routeInfo['validSeqs']
        validSeqsList.append(validSeqs)

        validSeqsGraph = routeInfo['validSeqsGraph']
        validSeqsGraphList.append(validSeqsGraph)

        routeImageListFile = os.path.join(imagePath, routeNameInDict + imageListSuffix + '.txt')
        subListFiles.append(routeImageListFile)

        routeDBFile = os.path.join(dbPath, routeNameInDict + imageListSuffix + '.db')
        subDBFiles.append(routeDBFile)

        routeModelPath = os.path.join(modelPath, routeNameInDict + imageListSuffix)
        subModelPaths.append(routeModelPath)

    routePairName = routeNameInDictList[0]
    for i in range(1, len(routeNameInDictList)):
        routePairName = routePairName + '_' + routeNameInDictList[i]

    # merge image lists and databses of seqs
    imageListFile = os.path.join(imagePath, routePairName + imageListSuffix + '.txt')
    dbFile = os.path.join(dbPath, routePairName + imageListSuffix + '.db')
    mergeImageListFile(subListFiles, imageListFile)
    mergeDBFile(subDBFiles, dbFile, colmapPath=colmapPath)

    # create match list for routePair
    matchListFile = os.path.join(dbPath, routePairName + imageListSuffix + '_match.txt')
    validSeqsGraphRoutePair = mergeTwoMatchGraph(validSeqsList[0], validSeqsList[1],
                                                 [], [])# 每个route内部的validSeqsGraph无需继承到validSeqsGraphRoutePair中

    createMatchList(imagePath, validSeqsGraphRoutePair, matchListFile, imageListSuffix=imageListSuffix)

    #routePair SFM
    routePairModelPath = os.path.join(modelPath, routePairName + imageListSuffix)
    if os.path.exists(routePairModelPath):
        shutil.rmtree(routePairModelPath)
    os.mkdir(routePairModelPath)

    inputModelPath = getOneValidModelPath([subModelPaths[0]])

    run_route_sfm(colmapPath, dbFile, imagePath, imageListFile, matchListFile, routePairModelPath,
                  matcher_min_num_inliers=matcher_min_num_inliers, mapper_min_num_matches=mapper_min_num_matches,
                  mapper_init_min_num_inliers=mapper_init_min_num_inliers,
                  mapper_abs_pose_min_num_inliers=mapper_abs_pose_min_num_inliers,
                  mapper_input_model_path=inputModelPath,
                  skip_feature_extractor=skip_feature_extractor,
                  mapper_fix_existing_images=mapper_fix_existing_images)

    return

def runAllRoutesSFM(colmapPath, imagePath, dbPath, modelPath, routeInfoList,
                    matcher_min_num_inliers=15, mapper_min_num_matches=45,
                    mapper_init_min_num_inliers=200, mapper_abs_pose_min_num_inliers=60, skip_feature_extractor=True,
                    imageListSuffix=''):
    for routeInfo in routeInfoList:
        runRouteSFM(colmapPath, imagePath, dbPath, modelPath, routeInfo,
                    matcher_min_num_inliers=matcher_min_num_inliers, mapper_min_num_matches=mapper_min_num_matches,
                    mapper_init_min_num_inliers=mapper_init_min_num_inliers,
                    mapper_abs_pose_min_num_inliers=mapper_abs_pose_min_num_inliers,
                    skip_feature_extractor=skip_feature_extractor,
                    imageListSuffix=imageListSuffix)

    return

def runAllRoutePairsSFM(colmapPath, imagePath, dbPath, modelPath, routePairInfoList,
                    matcher_min_num_inliers=15, mapper_min_num_matches=45,
                    mapper_init_min_num_inliers=200, mapper_abs_pose_min_num_inliers=60, skip_feature_extractor=True,
                    imageListSuffix='', useMergedSFMPlus=False, merger_max_reproj_error=64, merger_min_2d_inlier_percent=0.3,
                    mapper_fix_existing_images=0):
    for routePairInfo in routePairInfoList:
        if useMergedSFMPlus:
            runRoutePairSFMPlus(colmapPath, imagePath, dbPath, modelPath, routePairInfo,
                        matcher_min_num_inliers=matcher_min_num_inliers, mapper_min_num_matches=mapper_min_num_matches,
                        mapper_init_min_num_inliers=mapper_init_min_num_inliers,
                        mapper_abs_pose_min_num_inliers=mapper_abs_pose_min_num_inliers,
                        skip_feature_extractor=skip_feature_extractor,
                        imageListSuffix=imageListSuffix, merger_max_reproj_error=merger_max_reproj_error, merger_min_2d_inlier_percent=merger_min_2d_inlier_percent)
        else:
            runRoutePairSFM(colmapPath, imagePath, dbPath, modelPath, routePairInfo,
                            matcher_min_num_inliers=matcher_min_num_inliers,
                            mapper_min_num_matches=mapper_min_num_matches,
                            mapper_init_min_num_inliers=mapper_init_min_num_inliers,
                            mapper_abs_pose_min_num_inliers=mapper_abs_pose_min_num_inliers,
                            skip_feature_extractor=skip_feature_extractor,
                            imageListSuffix=imageListSuffix,
                            mapper_fix_existing_images=mapper_fix_existing_images)

    return



