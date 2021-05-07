# coding: utf-8
import os, sys
sys.path.append('../')
import shutil
from itertools import combinations
from colmap_process.colmap_export_geo import colmap_export_geo
from colmap_process.colmap_model_transformer import transformColmapModel

def readStrLines(file):
    lines = []
    with open(file) as fp:
        lines = fp.readlines()

    return lines

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

def copyAllColmapMaterials(srcName, dstName, imagesPath, dbPath, modelPath, imageListSuffix='', optionalMapper=False):
    srcImageList = os.path.join(imagesPath, srcName+imageListSuffix+'.txt')
    dstImageList = os.path.join(imagesPath, dstName+imageListSuffix+'.txt')
    shutil.copy(srcImageList, dstImageList)

    srcDB = os.path.join(dbPath, srcName + imageListSuffix + '.db')
    dstDB = os.path.join(dbPath, dstName + imageListSuffix + '.db')
    shutil.copy(srcDB, dstDB)

    if not optionalMapper:
        srcModel = os.path.join(modelPath, srcName + imageListSuffix)
        dstModel = os.path.join(modelPath, dstName + imageListSuffix)

        if os.path.exists(dstModel):
            shutil.rmtree(dstModel)

        os.mkdir(dstModel)

        copyModelFiles(srcModel, dstModel)

    return

def renameAllColmapMaterials(srcName, dstName, imagesPath, dbPath, modelPath, imageListSuffix=''):
    srcImageList = os.path.join(imagesPath, srcName+imageListSuffix+'.txt')
    dstImageList = os.path.join(imagesPath, dstName+imageListSuffix+'.txt')
    if os.path.isfile(srcImageList):
        os.rename(srcImageList, dstImageList)

    srcDB = os.path.join(dbPath, srcName + imageListSuffix + '.db')
    dstDB = os.path.join(dbPath, dstName + imageListSuffix + '.db')
    if os.path.isfile(srcDB):
        os.rename(srcDB, dstDB)
    
    srcModel = os.path.join(modelPath, srcName + imageListSuffix)
    dstModel = os.path.join(modelPath, dstName + imageListSuffix)

    if os.path.isdir(srcModel):
        os.rename(srcModel, dstModel)

    return

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

def run_model_orientation_aligner(colmap_exe, image_path, input_path, output_path, max_image_size=1024,
                                  rList=None, tList=None):
    run_str = colmap_exe + ' model_orientation_aligner' + \
              ' --image_path ' + image_path + \
              ' --input_path ' + input_path + \
              ' --output_path ' + output_path + \
              ' --max_image_size ' + str(max_image_size)

    print(run_str)
    os.system(run_str)

    #transform model
    if (not(rList==None)) and (not tList==None):
        transformColmapModel(output_path, output_path, rList=rList, tList=tList)

    # export geos.txt
    colmap_export_geo(output_path, [1, 0, 0, 0, 1, 0, 0, 0, 1])

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

def matchOrdinaryAndCharucoSeqs(ordinarySeqs, charucoSeqsList):
    seqsGraph = []

    seqsGraphOrdinary = matchSeqsExhaustive(ordinarySeqs, matchSelf=True)
    for seqs in charucoSeqsList:
        seqsGraph += matchSeqsExhaustive(seqs, matchSelf=True)
    
    for charucoSeqs in charucoSeqsList:
        seqsGraph += mergeTwoMatchGraph(charucoSeqs, ordinarySeqs, [], [])

    seqsGraphFull = seqsGraphOrdinary + seqsGraph.copy()

    return seqsGraph, seqsGraphFull

def matchSeqsExhaustive(seqList, matchSelf=False):
    outGraph = []

    combList = list(combinations(seqList, 2))
    for comb in combList:
        outGraph.append(list(comb))

    if matchSelf:
        for seq in seqList:
            outGraph.append([seq, seq])

    return outGraph

def matchTwoImageList(imageLsit1, imageList2):
    matchedList = []
    for image1 in imageLsit1:
        for image2 in imageList2:
            matchedList.append(image1 + ' ' + image2)

    return matchedList

def getImageListFileForMatch(imagePath, nodeName, imageListSuffix=''):
    tarFile = os.path.join(imagePath, nodeName + '_forMatch' + imageListSuffix + '.txt')

    if not os.path.isfile(tarFile):
        tarFile = os.path.join(imagePath, nodeName + imageListSuffix + '.txt')

    return tarFile

def createMatchList(imagePath, validSeqsGraph, matchListFile, imageListSuffix=''):
    matchList = []
    for edge in validSeqsGraph:
        imageList1 = readImageList(getImageListFileForMatch(imagePath, edge[0], imageListSuffix=imageListSuffix))
        imageList2 = readImageList(getImageListFileForMatch(imagePath, edge[1], imageListSuffix=imageListSuffix))

        matchList = matchList + matchTwoImageList(imageList1, imageList2)

    writeStrList(matchList, matchListFile)

def getSubFolders(folderPath):
    subNames= []
    if os.path.isdir(folderPath):
        subNames = os.listdir(folderPath)

    subFolders = []
    for name in subNames:
        fullFile = os.path.join(folderPath, name)
        if os.path.isdir(fullFile):
            subFolders.append(fullFile)

    return subFolders

def getOneValidModelPath(modelPaths):
    tarPath = ""
    for modelPath in modelPaths:
        subFolders = getSubFolders(modelPath)

        if len(subFolders) == 1:
            tarPath = subFolders[0]
            break

    return tarPath

def checkAndDeleteFiles(fileList):
    for file in fileList:
        if os.path.exists(file):
            if os.path.isdir(file):
                shutil.rmtree(file)
            else:
                os.remove(file)
    return

def cleanColmapMaterialsByNames(names, imagesPath, dbPath, modelPath, imageListSuffix=''):
    deleteList = []
    for name in names:
        imageListFile = os.path.join(imagesPath, name+imageListSuffix+".txt")
        deleteList.append(imageListFile)

        dbFile = os.path.join(dbPath, name+imageListSuffix+".db")
        deleteList.append(dbFile)

        matchListFile = os.path.join(dbPath, name+imageListSuffix+"_match.txt")
        deleteList.append(matchListFile)

        modelDirPath = os.path.join(modelPath, name+imageListSuffix)
        deleteList.append(modelDirPath)

    checkAndDeleteFiles(deleteList)

    return

def checkExistence(dirNameList, imagePath):
    exceptNameList = []

    for name in dirNameList:
        if not os.path.isdir(os.path.join(imagePath, name)):
            exceptNameList.append(name)

    if len(exceptNameList)>0:
        print('the following subDirs do not exist in ' + imagePath + ':\n')
        for name in exceptNameList:
            print(name)
        
        raise Exception("imageSubDirs missing, check the print info above.")

    return