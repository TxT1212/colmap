# coding: utf-8
import os
import json
from itertools import combinations

class mapNode:
    def __init__(self, name="", inheritNames=[], type="", parent="", cameraModel="SIMPLE_RADIAL", focalLenFactor=1.2, intervalForMatch=1):
        self.name = name
        self.inheritNames = inheritNames.copy()
        self.type = type
        self.idx = -1
        self.subNodes = []
        self.edges = []
        self.edgeHooks = []
        self.edgeHooksInName = []
        self.parent = parent
        self.validSeqs = []
        self.validSeqExts = []
        self.validSeqsGraph = []
        self.charucoSeqs = []
        self.charucoRoutes = []
        self.shouldRunMapper = True
        self.mergeOrders = []
        self.interconnected = True
        self.cameraModel = cameraModel
        self.focalLenFactor = focalLenFactor
        self.intervalForMatch = intervalForMatch
        self.startMergeId=-1
        self.importMergeOrders=False
        self.runBaseRecon = True
        self.runDBClean=True
        self.runFeatureExtractor=True
        self.runMatcher=True
        self.runMapper='neutral'
        self.runScaleRec='neutral'
        self.runGravityRec='neutral'
        self.runCharucoDelete='neutral'

def writeStrList(strList, outFile):
    with open(outFile, 'w') as fp:
        for strEle in strList:
            fp.write(strEle + '\n')
    
    return

def getNodeWithMaxNumNeighbors(tarNodeList, nodesNeighbors, nodesPicks):
    outNode = ""
    maxNumNeighbors = -1

    for node in tarNodeList:
        numNeighbors = len(nodesNeighbors[node])
        if (not nodesPicks[node]) and (numNeighbors>maxNumNeighbors):
            outNode = node
            maxNumNeighbors = numNeighbors

    return outNode

def getMergeOrders(nodes, edges, interconnected=True):
    orders = []

    if not interconnected:
        orders = nodes.copy()
    else:
        # create nodesNeighbors dict
        nodesNeighbors = {}
        nodesPicks = {}
        for node in nodes:
            nodesNeighbors[node] = []
            nodesPicks[node] = False

        # get neighbors for each node
        for edge in edges:
            nodesNeighbors[edge[0]].append(edge[1])
            nodesNeighbors[edge[1]].append(edge[0])

        tarNodeList = list(nodesNeighbors.keys())
        while len(orders) < len(nodes):
            pickNode = getNodeWithMaxNumNeighbors(tarNodeList, nodesNeighbors, nodesPicks)

            if pickNode=='':
                raise Exception('pickNode is empty, meaning the scene graph of current node is broken.\n')

            orders.append(pickNode)
            nodesPicks[pickNode] = True
            tarNodeList = getAssociatedNodes(orders, nodesNeighbors)

    return orders

def splitGraph(nodes, edges):
    # {'baseName':'', 'augEdges':[['', '']], 'augNodes': [''], 'augName':, ''}
    augmentedNodes = []
    edgesPicks = {}

    for i in range(len(edges)):
        edgesPicks[i] = False
    
    for node in nodes:
        augNode = {}
        augNode['baseName'] = node
        augNode['augEdges'] = []
        augNode['augNodes'] = []
        augNode['augName'] = augNode['baseName'] + '_AugTo'

        asscoEdges, edgeIdxs, nodeIdxInEdges = getAssocEgedsByOneNodeName(edges, node)

        for edgeIdx in edgeIdxs:
            if not edgesPicks[edgeIdx]:
                augNode['augEdges'].append(edges[edgeIdx])
                edgesPicks[edgeIdx] = True

                edgeTmp = edges[edgeIdx].copy()
                edgeTmp.remove(node)
                augNode['augNodes'].append(edgeTmp[0])
                augNode['augName'] = augNode['augName'] + '_' + edgeTmp[0]
        
        augmentedNodes.append(augNode)

    return augmentedNodes

def getAssociatedNodes(pickedNodes, nodesNeighbors):
    assocNodes = []

    for pickNode in pickedNodes:
        neighbors = nodesNeighbors[pickNode]

        for node in neighbors:
            if not (node in assocNodes):
                assocNodes.append(node)

    return assocNodes

def matchPair(tarEdge, edge):
    flag = False
    tarNode1IdxInMatchedEdge = -1
    tarNode2IdxInMatchedEdge = -1

    if (tarEdge[0]==edge[0]) and (tarEdge[1]==edge[1]):
        flag = True
        tarNode1IdxInMatchedEdge = 0
        tarNode2IdxInMatchedEdge = 1
    elif (tarEdge[0]==edge[1]) and(tarEdge[1]==edge[0]):
        flag = True
        tarNode1IdxInMatchedEdge = 1
        tarNode2IdxInMatchedEdge = 0

    return flag, tarNode1IdxInMatchedEdge, tarNode2IdxInMatchedEdge

def getMatchedEdge(tarEdge, edgeList):
    isMatch = False
    tarNode1IdxInMatchedEdge = -1
    tarNode2IdxInMatchedEdge = -1

    matchedEdgeIdx = -1
    for i in range(len(edgeList)):
        edge = edgeList[i]
        isMatch, tarNode1IdxInMatchedEdge, tarNode2IdxInMatchedEdge = matchPair(tarEdge, edge)
        if isMatch:
            matchedEdgeIdx = i
            break

    return matchedEdgeIdx, tarNode1IdxInMatchedEdge, tarNode2IdxInMatchedEdge

def parseSceneNode(item, sceneTree):
    node = mapNode()
    node.name = item["name"]
    node.idx = item["idx"]
    node.type = item["type"]
    node.inheritNames.append(item["name"])

    memberIndices = item["members"]["indices"]
    memberNodes = {}
    for i in range(len(memberIndices)):
        memberNodes[memberIndices[i][0]] = memberIndices[i][1]
        node.subNodes.append(memberIndices[i][1])

    memberEdges = item["members"]["edges"]
    for i in range(len(memberEdges)):
        node.edges.append([memberNodes[memberEdges[i][0]], memberNodes[memberEdges[i][1]]])
        node.edgeHooks.append([])

    sceneTree[0][node.name] = node

def sameEdges(edges1, edges2):
    flag = True

    if len(edges1) == len(edges2):
        for edge in edges1:
            if not (edge in edges2):
                flag = False
                break
    else:
        flag = False

    return flag

def parsePlaceNode(item, sceneTree):
    node = mapNode()
    node.name = item["name"]
    node.idx = item["idx"]
    node.type = item["type"]
    node.parent = item["parent"]
    node.inheritNames = sceneTree[0][item["parent"]].inheritNames.copy()
    node.inheritNames.append(item["name"])

    # if curNode's edges are interconnected
    if 'interconnected' in item.keys():
        node.interconnected = item['interconnected']

    memberIndices = item["members"]["indices"]
    memberNodes = {}
    for i in range(len(memberIndices)):
        memberNodes[memberIndices[i][0]] = memberIndices[i][1]
        node.subNodes.append(memberIndices[i][1])

        #create route node
        routeNode = mapNode()
        routeNode.name = memberIndices[i][1]
        routeNode.idx = memberIndices[i][0]
        routeNode.type = "route"
        routeNode.parent = item["name"]
        routeNode.inheritNames = node.inheritNames.copy()
        routeNode.inheritNames.append(memberIndices[i][1])
        
        if len(memberIndices[i]) > 2:
            routeNode.cameraModel = memberIndices[i][2]

            if len(memberIndices[i]) > 3:
                routeNode.focalLenFactor = memberIndices[i][3]

                if len(memberIndices[i]) > 4:
                    routeNode.intervalForMatch = memberIndices[i][4]

        routeNodeDictName = routeNode.parent + "_" + routeNode.name
        sceneTree[2][routeNodeDictName] = routeNode

    memberEdges = item["members"]["edges"]
    for i in range(len(memberEdges)):
        node.edges.append([memberNodes[memberEdges[i][0]], memberNodes[memberEdges[i][1]]])

    # get charuco route info
    node.charucoRoutes = []
    if "charuco_ids" in item["members"].keys():
        memberCharucoIds = item["members"]["charuco_ids"]
        for i in range(len(memberCharucoIds)):
            routeName = memberNodes[memberCharucoIds[i]]
            if not (routeName in node.charucoRoutes):
                node.charucoRoutes.append(routeName)
    elif "charuco_pairs" in item["members"].keys():
        memberCharucoPairs = item["members"]["charuco_pairs"]
        for i in range(len(memberCharucoPairs)):
            routeName = memberNodes[memberCharucoPairs[i][0]]
            if not (routeName in node.charucoRoutes):
                node.charucoRoutes.append(routeName)
    else:
         raise Exception('no either charuco_ids nor charuco_pairs is in place item')

    sceneTree[1][node.name] = node

    # upload the neighbors edge info to parent node
    neighbors = item["neighbors"]
    for neigh in neighbors:
        neighPair = [item["name"], neigh["name"]]
        edgeIdx, curNodeIdxInEdge, neighNodeIdxInEdge = getMatchedEdge(neighPair, sceneTree[0][item["parent"]].edges)
        if edgeIdx<0:
            raise Exception("could not find matched edge in parent node")
        
        hookEdges = neigh["edges"]
        if len(sceneTree[0][item["parent"]].edgeHooks[edgeIdx]) == 0:  
            for edge in hookEdges:
                # TODO: delete duplicated hook append
                hookEdge = [edge[curNodeIdxInEdge], edge[neighNodeIdxInEdge]]
                if not (hookEdge in sceneTree[0][item["parent"]].edgeHooks[edgeIdx]):
                    sceneTree[0][item["parent"]].edgeHooks[edgeIdx].append(hookEdge)
        else:
            edgesTemp = []
            for edge in hookEdges:
                hookEdge = [edge[curNodeIdxInEdge], edge[neighNodeIdxInEdge]]
                if not (hookEdge in edgesTemp):
                    edgesTemp.append(hookEdge)

            if not sameEdges(edgesTemp, sceneTree[0][item["parent"]].edgeHooks[edgeIdx]):
                exceptStr = 'route associations between ' + neighPair[0] + ' and ' + neighPair[1] + \
                    ' is not consistent with the previous read setting.\n' + \
                    'Check if there exist duplicated and inconsistent settings in scene_graph.json.'
                raise Exception(exceptStr)
    
    return

def getValidSeqsForRouteNode(projPath, projName, batchName, node):
    inheritNamesList = node.inheritNames
    seqName = projName + "_" + inheritNamesList[0] + "_" + batchName + "_" +\
              inheritNamesList[1] + "_" + inheritNamesList[2] + "_0001"

    node.validSeqs.append(seqName)

    node.subNodes.append(seqName)

    for seq1 in node.validSeqs:
        for seq2 in node.validSeqs:
            if seq1 == seq2: # 序列内部的match关系由colmap_seq_sfm定义（当前序列内使用的是sequential match）
                continue

            node.validSeqsGraph.append([seq1, seq2])

    return

def getValidSeqsFromRouteJson(configPath, node):
    routeJsonName = node.inheritNames[-3] + '_' + node.inheritNames[-2] + '_' + node.inheritNames[-1]
    routeJsonFile = os.path.join(configPath, routeJsonName + '.json')
    if not os.path.isfile(routeJsonFile):
        raise Exception('routeJsonFile does not exist: %s\n' % routeJsonFile)

    with open(routeJsonFile, 'r', encoding='UTF-8') as fp:
        jsonValue = json.load(fp)

    validSeqFiles = jsonValue['validSeqs']

    for seqFile in validSeqFiles:
        seqName, ext = os.path.splitext(seqFile)
        node.validSeqs.append(seqName)
        node.subNodes.append(seqName)
        node.validSeqExts.append(ext)

    combs = combinations(node.validSeqs, 2)
    for comb in combs:
        node.validSeqsGraph.append(list(comb))

    return

def getAllCharucoSeqs(sceneTree):
    # place node
    charucoSeqs = []
    for key in sceneTree[1].keys():
        placeNode = sceneTree[1][key]
        charucoSeqs += placeNode.charucoSeqs
    
    return charucoSeqs

def identifyCharucoSeqs(tarSeqs, sceneTree):
    charucoSeqsFlags = {}
    allCharucoSeqs = getAllCharucoSeqs(sceneTree)

    for seq in tarSeqs:
        if seq in allCharucoSeqs:
            charucoSeqsFlags[seq] = True
        else:
            charucoSeqsFlags[seq] = False

    return charucoSeqsFlags

def getSeqs(sceneTree):
    # place node
    for key in sceneTree[1].keys():
        placeNode = sceneTree[1][key]
        placeNode.validSeqs = []
        for subNodeName in placeNode.subNodes:
            subNodeNameInDict = getSubNodeNameInDict(placeNode.inheritNames, placeNode.type, subNodeName)
            placeNode.validSeqs += sceneTree[2][subNodeNameInDict].validSeqs

    # scene node
    for key in sceneTree[0].keys():
        sceneNode = sceneTree[0][key]

        sceneNode.validSeqs = []
        for subNodeName in sceneNode.subNodes:
            subNode = sceneTree[1][subNodeName]
            sceneNode.validSeqs += subNode.validSeqs

    return

def getCharucoSeqsListInPlace(placeNode, serialTopology):
    charucoSeqsList = []
    for charucoRoute in placeNode.charucoRoutes:
        charucoRouteNameInDict = getSubNodeNameInDict(placeNode.inheritNames, placeNode.type, charucoRoute)
        charucoSeqsList.append(serialTopology[charucoRouteNameInDict].validSeqs)

    return charucoSeqsList

def serializeList(oriList):
    outList = []
    for subList in oriList:
        outList += subList
    
    return outList

def getOrdinaryAndCharucoSeqs(mapNode, serialTopology):
    ordinarySeqs = []
    charucoSeqsList = []

    if mapNode.type == 'route':
        ordinarySeqs += mapNode.validSeqs
    elif mapNode.type == 'place':
        charucoSeqsList += getCharucoSeqsListInPlace(mapNode, serialTopology)

        charucoSeqs = serializeList(charucoSeqsList)
        for seq in mapNode.validSeqs:
            if not seq in charucoSeqs:
                ordinarySeqs.append(seq)

    elif mapNode.type == 'scene':
        for placeNodeName in mapNode.subNodes:
            charucoSeqsList += getCharucoSeqsListInPlace(serialTopology[placeNodeName], serialTopology)
        
        charucoSeqs = serializeList(charucoSeqsList)
        for seq in mapNode.validSeqs:
            if not seq in charucoSeqs:
                ordinarySeqs.append(seq)
    else:
        raise Exception('unsupported node type: ' + mapNode.type)

    return ordinarySeqs, charucoSeqsList


def getCharucoSeqs(sceneTree):
    # place node
    for key in sceneTree[1].keys():
        placeNode = sceneTree[1][key]
        charucoRoutes = placeNode.charucoRoutes

        placeNode.charucoSeqs = []
        for charucoRoute in charucoRoutes:
            routeNameInDict = placeNode.name + '_' + charucoRoute
            routeNode = sceneTree[2][routeNameInDict]

            placeNode.charucoSeqs += routeNode.validSeqs

    # scene node
    for key in sceneTree[0].keys():
        sceneNode = sceneTree[0][key]

        sceneNode.charucoSeqs = []
        for subNodeName in sceneNode.subNodes:
            subNode = sceneTree[1][subNodeName]
            sceneNode.charucoSeqs += subNode.charucoSeqs

    return

def saveCameraConfig(savePath, serialTopology):
    if not os.path.isdir(savePath):
        os.makedirs(savePath)
    
    for key, node in serialTopology.items():
        configFile = os.path.join(savePath, key+'.json')

        configDict = {}
        configDict['cameraModel'] = node.cameraModel
        configDict['focalLenFactor'] = node.focalLenFactor
        
        with open(configFile, 'w') as fp:
            json.dump(configDict, fp, indent=4)

    return

def saveImageListForMatch(savePath, serialTopology, imageListSuffix=''):
    for key, node in serialTopology.items():
        if node.intervalForMatch > 1:
            oriImageListFile = os.path.join(savePath, key + imageListSuffix + '.txt')
            sampledImageListFile = os.path.join(savePath, key + '_forMatch' + imageListSuffix + '.txt')

            if os.path.isfile(oriImageListFile):
                lines = []
                with open(oriImageListFile) as fp:
                    lines = fp.readlines()

                with open(sampledImageListFile, 'w') as fp:
                    for i in range(0, len(lines), int(node.intervalForMatch+0.5)):
                        fp.write(lines[i])
        else:
            sampledImageListFile = os.path.join(savePath, key + '_forMatch' + imageListSuffix + '.txt')
            if os.path.isfile(sampledImageListFile):
                os.remove(sampledImageListFile)

    return

def getSceneTopology(projPath, projName=None, batchName=None, sceneGraphName='scene_graph',
                     configPath=None, validSeqsFromJson=False, suppleGraphName='', supplePlaces=[],
                     imagesDir=None, taskPath=None):
    if validSeqsFromJson:
        if not os.path.isdir(configPath):
            raise Exception('validSeqsFromJson=True, but can not find configPath: %s' % configPath)
    else:
        if (projName==None) or (batchName==None):
            raise Exception('validSeqsFromJson=False, but no projName or batchName inputted.')

    sceneGraphJson = os.path.join(projPath, sceneGraphName + ".json")
    with open(sceneGraphJson, 'r', encoding='UTF-8') as f:
        sceneValues = json.load(f)

    sceneItems = sceneValues["Items"]

    # if there exists suppleGraphFile, running with map supplement mode
    suppleGraphFile = os.path.join(projPath, suppleGraphName+'.json')
    if os.path.isfile(suppleGraphFile):
        with open(suppleGraphFile, 'r', encoding='UTF-8') as f:
            sceneValuesSupple = json.load(f)
            sceneItemsSupple = sceneValuesSupple["Items"]
        
        suppleSceneItem = None
        for item in sceneItemsSupple:
            if item["type"] == 'scene':
                suppleSceneItem = item
            elif item['type'] == 'place':
                supplePlaces.append(item['name'])
            else:
                raise Exception('unsupported item type: ' + item['type'])

        for i in range(len(sceneItems)):
            item = sceneItems[i]
            if (item["type"] == 'scene') and (item['name'] == suppleSceneItem['name']):
                sceneItems.pop(i)
                break
        
        sceneItems = sceneItemsSupple[0:1] + sceneItems + sceneItemsSupple[1:]
  
    # build scene tree
    # parse each item
    sceneTree = []
    sceneTree.append({}) # scenes
    sceneTree.append({}) # palces
    sceneTree.append({}) # routes
    sceneTree.append({}) # sequences

    for item in sceneItems:
        if item["type"] == "scene":
            parseSceneNode(item, sceneTree)
        elif item["type"] == "place":
            parsePlaceNode(item, sceneTree)

    # merge validSeqs and validSeqsGraph
    # get validSeqs for route nodes
    for key in sceneTree[2].keys():
        routeNode = sceneTree[2][key]
        if validSeqsFromJson:
            getValidSeqsFromRouteJson(configPath, routeNode)
        else:
            getValidSeqsForRouteNode(projPath, projName, batchName, routeNode)

        # build sequence nodes
        for seqs in routeNode.validSeqs:
            node = mapNode(name=seqs, inheritNames=routeNode.inheritNames + [seqs],
                           type='sequence', parent=routeNode.name,
                           cameraModel=routeNode.cameraModel, focalLenFactor=routeNode.focalLenFactor,
                           intervalForMatch=routeNode.intervalForMatch)
            sceneTree[3][seqs] = node

    #get validSeqs for each place and scene
    getSeqs(sceneTree)

    # get charucoSeqs for each place and scene
    getCharucoSeqs(sceneTree)

    # get mergeOrders of scene node and palce node
    getMergeOrdersOfSubNodes(sceneTree[0])
    getMergeOrdersOfSubNodes(sceneTree[1])

    # set shouldRunMapper of non-first subnodes as False according to node.mergeOrders
    setChildNodesShouldRunMapper(sceneTree)

    # set camera settings for each sequence
    if not (imagesDir==None):
        saveCameraConfig(os.path.join(projPath, imagesDir, 'cameraConfig'), sceneTree[3])

    # save all node namesInDict and baseReconConfig.json
    if not (taskPath==None):
        savePath = os.path.join(taskPath, sceneGraphName)

        if not os.path.isdir(savePath):
            os.makedirs(savePath)

        nodeListFile = os.path.join(savePath, 'all_node_names_in_scene.txt')
        AllNodeNamesInDict = list(sceneTree[3].keys()) + list(sceneTree[2].keys()) + list(sceneTree[1].keys()) +list(sceneTree[0].keys())
        writeStrList(AllNodeNamesInDict, nodeListFile)

        baseReconConfigFile = os.path.join(savePath, 'baseReconConfig.json')
        saveBaseReconConfig(baseReconConfigFile, getSerializedMapNodes(sceneTree))

    return sceneTree

def saveBaseReconConfig(configFile, serialTopology):
    configDict = {'nodeConfig': []}

    for key, node in serialTopology.items():
        nodeConfig = {}
        nodeConfig['nodeNameInDict'] = key
        nodeConfig['startMergeId'] = node.startMergeId
        nodeConfig['importMergeOrders'] = node.importMergeOrders
        nodeConfig['runBaseRecon'] = node.runBaseRecon
        nodeConfig['sfmConfig'] = {}
        nodeConfig['sfmConfig']['runDBClean'] = node.runDBClean
        nodeConfig['sfmConfig']['runFeatureExtractor'] = node.runFeatureExtractor
        nodeConfig['sfmConfig']['runMatcher'] = node.runMatcher
        nodeConfig['sfmConfig']['note'] = '以下参数必须设置为以下三个值之一, negative:不运行；positive:运行；neutral:由算法自行决定'
        nodeConfig['sfmConfig']['runMapper'] = node.runMapper
        nodeConfig['sfmConfig']['note'] = '以下参数必须设置为以下三个值之一, negative:不运行；positive:运行；neutral:由算法自行决定'
        nodeConfig['runScaleRec'] = node.runScaleRec
        nodeConfig['runGravityRec'] = node.runGravityRec
        nodeConfig['runCharucoDelete'] = node.runCharucoDelete

        configDict['nodeConfig'].append(nodeConfig)

    with open(configFile, 'w') as fp:
            json.dump(configDict, fp, indent=4)

    return

def getAllSeqListInRouteNodes(routeNodes, routeNamesInDict=[]):
    seqList = []
    for key in routeNodes.keys():

        if (len(routeNamesInDict)>0) and (not (key in routeNamesInDict)):
            continue
         
        routeNode = routeNodes[key]
        seqList = seqList + routeNode.validSeqs

    return seqList

def getAllRouteListInRouteNodes(routeNodes):
    routeList = []
    for key in routeNodes.keys():
        routeNode = routeNodes[key]

        routeInfo = {}
        routeInfo["inheritNames"] = routeNode.inheritNames
        routeInfo["type"] = routeNode.type
        routeInfo["validSeqs"] = routeNode.validSeqs
        routeInfo["validSeqsGraph"] = routeNode.validSeqsGraph
        routeInfo['shouldRunMapper'] = routeNode.shouldRunMapper
        routeInfo['mergeOrders'] = routeNode.mergeOrders.copy()

        routeList.append(routeInfo)

    return routeList

def getAllRoutePairListInPlaceNodes(sceneTopology):
    routePairList = []

    for key in sceneTopology[1].keys():
        palceNode = sceneTopology[1][key]

        for edge in palceNode.edges:
            routePair = []
            for routeName in edge:
                routeNodeDictName = palceNode.name + "_" + routeName
                routeNode = sceneTopology[2][routeNodeDictName]

                routeInfo = {}
                routeInfo["inheritNames"] = routeNode.inheritNames
                routeInfo["type"] = routeNode.type
                routeInfo["validSeqs"] = routeNode.validSeqs
                routeInfo["validSeqsGraph"] = routeNode.validSeqsGraph

                routePair.append(routeInfo)

            routePairList.append(routePair)

    return routePairList

def getOnlyNamesFromNodesInfo(nodesInfoList):
    nodesNamesList = []

    for infoList in nodesInfoList:
        names = []
        for info in infoList:
            if type(info) is str:
                names.append(info)
            elif type(info) is dict:
                name = getNodeNameInDict(info['inheritNames'], info['type'])
                names.append(name)
            elif type(info) is list:
                namePairs = []
                for nodeInfo in info:
                    namePairs.append(getNodeNameInDict(nodeInfo['inheritNames'], nodeInfo['type']))

                names.append(namePairs)
            else:
                raise Exception('unsupported info type')

        nodesNamesList.append(names)

    return nodesNamesList

def getAllRoutePairListInSceneNodes(sceneTopology):
    routePairList = []
    for key in sceneTopology[0].keys():
        sceneNode = sceneTopology[0][key]

        for i in range(len(sceneNode.edges)):
            edge = sceneNode.edges[i]
            for hook in sceneNode.edgeHooks[i]:
                routePair = []
                for j in range(2):
                    placeName = edge[j]
                    routeIdx = hook[j]

                    placeNode = sceneTopology[1][placeName]
                    routeName = placeNode.subNodes[routeIdx-1]

                    routeNodeDictName = placeName + "_" + routeName
                    routeNode = sceneTopology[2][routeNodeDictName]

                    routeInfo = {}
                    routeInfo["inheritNames"] = routeNode.inheritNames
                    routeInfo["type"] = routeNode.type
                    routeInfo["validSeqs"] = routeNode.validSeqs
                    routeInfo["validSeqsGraph"] = routeNode.validSeqsGraph

                    routePair.append(routeInfo)

                routePairList.append(routePair)

    return routePairList

def matchedEdge(srcEdge, tarEdge):
    flag = False
    if (srcEdge[0] in tarEdge) and (srcEdge[1] in tarEdge):
        flag = True

    return flag

def getMatchedEdgeIdx(tarEdge, edgeList):
    idx = -1
    for i in range(len(edgeList)):
        edge = edgeList[i]
        if matchedEdge(tarEdge, edge):
            idx = i
            break
    return idx

def getSceneHooksByEdge(sceneName, edges, serialTopology):
    outPairs = []

    sceneNode = serialTopology[sceneName]
    for edge in edges:
        edgeIdx = getMatchedEdgeIdx(edge, sceneNode.edges)
        if edgeIdx < 0:
            continue

        for hook in sceneNode.edgeHooks[edgeIdx]:
            routePair = []
            for j in range(2):
                placeName = edge[j]
                routeIdx = hook[j]

                placeNode = serialTopology[placeName]
                routeName = placeNode.subNodes[routeIdx - 1]

                routeNodeDictName = placeName + "_" + routeName
                routePair.append(routeNodeDictName)

            outPairs.append(routePair)

    return outPairs

def getAllNodesInfoInSerialTopology(serialTopology):
    nodelInfoList = []

    for key in serialTopology.keys():
        node = serialTopology[key]

        nodeInfo = {}
        nodeInfo["inheritNames"] = node.inheritNames
        nodeInfo["type"] = node.type
        nodeInfo['shouldRunMapper'] = node.shouldRunMapper
        nodeInfo['mergeOrders'] = node.mergeOrders.copy()

        nodelInfoList.append(nodeInfo)

    return nodelInfoList

def getNodeInfo(nodeNameInDict, serialTopology):
    nodeType = 'undefined'
    nodeInfo = {}

    if nodeNameInDict in serialTopology.keys():
        node = serialTopology[nodeNameInDict]
        nodeType = node.type

        if node.type == 'sequence':
            nodeInfo = nodeNameInDict
        else:
            nodeInfo = {}
            nodeInfo["inheritNames"] = node.inheritNames
            nodeInfo["type"] = node.type
            nodeInfo["validSeqs"] = node.validSeqs
            nodeInfo["validSeqsGraph"] = node.validSeqsGraph
    else:
        print(nodeNameInDict + ' is not a valid node in topology.\n')

    return nodeType, nodeInfo

def getTopologyCheckTasksSpecified(strLines, sceneTopology, seperator=' '):
    serialTopology = getSerializedMapNodes(sceneTopology)
    seqToCheckList = []
    routeToCheckList = []
    routePairToCheckList = []

    for line in strLines:
        line = line.strip()

        if len(line)<=0 or line[0]=='#':
            continue

        lineParts = line.split(seperator)

        if len(lineParts) == 1:
            nodeType, nodeInfo = getNodeInfo(lineParts[0], serialTopology)

            if nodeType == 'sequence':
                seqToCheckList.append(nodeInfo)
            elif nodeType == 'route':
                routeToCheckList.append(nodeInfo)
            else:
                print('unsupported node type of %s for now' % lineParts[0])
        else:
            allRoutePairInfoList = getRoutePairCheckTasks(sceneTopology)
            matchedRoutePair = getMatchedRoutePair([lineParts[0], lineParts[1]], allRoutePairInfoList)

            if len(matchedRoutePair) == 2:
                routePairToCheckList.append(matchedRoutePair)
            else:
                print('sepcified routepair (%s, %s) does not match any routepair in scene graph.\n'
                      % (lineParts[0], lineParts[1]))

    return seqToCheckList, routeToCheckList, routePairToCheckList

def getMatchedRoutePair(namePair, routePairList):
    matchedRoutePair = []

    for pair in routePairList:
        inheritNames1 = pair[0]['inheritNames']
        routeName1 = inheritNames1[-2] + '_' + inheritNames1[-1]
        inheritNames2 = pair[1]['inheritNames']
        routeName2 = inheritNames2[-2] + '_' + inheritNames2[-1]

        if (routeName1 in namePair) and (routeName2 in namePair):
            matchedRoutePair = pair
            break

    return matchedRoutePair

def allCheckTasksPassed(taskList, checkStatusDict):
    flag = True

    for task in taskList:
        if (not (task in checkStatusDict.keys())) or (not (checkStatusDict[task] == 'success')):
            flag = False
            break
    return flag

def getFailedRoutesAndSeqs(routeList, routeStatusdict, sceneTopology):
    serialTopology = getSerializedMapNodes(sceneTopology)
    failedRouteList = []
    assocSeqsList = []

    for routeInfo in routeList:
        routeNameInDict = getNodeNameInDict(routeInfo['inheritNames'], routeInfo['type'])
        if not allCheckTasksPassed([routeNameInDict], routeStatusdict):
            failedRouteList.append(routeInfo)
            assocSeqsList += serialTopology[routeNameInDict].subNodes

    return failedRouteList, assocSeqsList

def filterTopologyCheckTasks(inputTaskList, previousTaskStatusDict, sceneTopology):
    serialTopology = getSerializedMapNodes(sceneTopology)
    outputTaskList = []
    deleteTaskNameList = []

    for task in inputTaskList:
        if type(task) is str:
            print('sequence check task does not need to be filtered.\n')
            continue
        elif type(task) is dict:
            if not (task['type']=='route'):
                raise Exception('expected type of this task is route, but here we found the actual type is ' + task['type'])

            nodeNameInDict = getNodeNameInDict(task['inheritNames'], task['type'])
            mapNode = serialTopology[nodeNameInDict]
            subNodesNamesIndDict = mapNode.subNodes

            if allCheckTasksPassed(subNodesNamesIndDict, previousTaskStatusDict):
                outputTaskList.append(task)
            else:
                print('delete %s in taskList due to failed subNodes.\n' % nodeNameInDict)
                deleteTaskNameList.append(nodeNameInDict)
        elif type(task) is list:
            nodeInfo1 = task[0]
            nodeInfo2 = task[1]
            if (not (nodeInfo1['type']=='route')) or (not (nodeInfo2['type']=='route')):
                raise Exception('expected types of this pair are both route, but it is actually not.')
            subNodesNamesIndDict=[]
            subNodesNamesIndDict.append(getNodeNameInDict(nodeInfo1['inheritNames'], nodeInfo1['type']))
            subNodesNamesIndDict.append(getNodeNameInDict(nodeInfo2['inheritNames'], nodeInfo2['type']))

            if allCheckTasksPassed(subNodesNamesIndDict, previousTaskStatusDict):
                outputTaskList.append(task)
            else:
                print('delete (%s, %s) in taskList due to failed subNodes.\n' % (subNodesNamesIndDict[0], subNodesNamesIndDict[1]))
                deleteTaskNameList.append(subNodesNamesIndDict[0]+'_'+subNodesNamesIndDict[1])
        else:
            raise Exception('unsupported info type of inputed task.')

    return outputTaskList, deleteTaskNameList

def getRoutePairCheckTasks(sceneTopology):
    # get routePairList to be checked in placeNodes
    routePairToCheckListInPlaces = getAllRoutePairListInPlaceNodes(sceneTopology)

    # get routePairList to be checked in sceneNodes
    routePairToCheckListInScenes = getAllRoutePairListInSceneNodes(sceneTopology)

    routePairToCheckList = routePairToCheckListInPlaces + routePairToCheckListInScenes

    return routePairToCheckList

def getTopologyCheckTasks(sceneTopology):
    # get seqList to be checked
    seqToCheckList = getAllSeqListInRouteNodes(sceneTopology[2])

    # get routeList to be checked
    routeToCheckList = getAllRouteListInRouteNodes(sceneTopology[2])

    # get routePairList to be checked in placeNodes
    routePairToCheckListInPlaces = getAllRoutePairListInPlaceNodes(sceneTopology)

    # get routePairList to be checked in sceneNodes
    routePairToCheckListInScenes = getAllRoutePairListInSceneNodes(sceneTopology)

    routePairToCheckList = routePairToCheckListInPlaces + routePairToCheckListInScenes

    return seqToCheckList, routeToCheckList, routePairToCheckList

def filterTopologyCheckTasksByPlaces(seqCheckList, routeCheckList, routePairCheckList, sceneTopology, tarPlaces):
    seqToCheckList = []
    routeToCheckList = []
    routePairToCheckList = []

    for seqName in seqCheckList:
        seqNode = sceneTopology[3][seqName]
        placeName = seqNode.inheritNames[-3]
        if placeName in tarPlaces:
            seqToCheckList.append(seqName)
    
    for routeInfo in routeCheckList:
        placeName = routeInfo['inheritNames'][-2]
        if placeName in tarPlaces:
            routeToCheckList.append(routeInfo)
    
    for routePairInfo in routePairCheckList:
        placeName1 = routePairInfo[0]['inheritNames'][-2]
        placeName2 = routePairInfo[1]['inheritNames'][-2]

        if (placeName1 in tarPlaces) or (placeName2 in tarPlaces):
            routePairToCheckList.append(routePairInfo)

    return seqToCheckList, routeToCheckList, routePairToCheckList

def setSubNodesShouldRunMapper(curLevelNode, nextLevelTopology, startIdx=0, flag=False):
    for i in range(startIdx, len(curLevelNode.mergeOrders)):
        subNodeNameInDict = getSubNodeNameInDict(curLevelNode.inheritNames, curLevelNode.type,
                                                   curLevelNode.mergeOrders[i])
        nextLevelTopology[subNodeNameInDict].shouldRunMapper = flag

    return

def setChildNodesShouldRunMapper(sceneTopology):
    # set default
    setAllNodesShouldRunMapper(getSerializedMapNodes(sceneTopology), True)

    # scene and palce
    for key in sceneTopology[0].keys():
        sceneNode = sceneTopology[0][key]
        setSubNodesShouldRunMapper(sceneNode, sceneTopology[1], startIdx=1, flag=False)

        for i in range(len(sceneNode.mergeOrders)):
            placeNodeNameInDict = getSubNodeNameInDict(sceneNode.inheritNames, sceneNode.type,
                                                     sceneNode.mergeOrders[i])
            placeNode = sceneTopology[1][placeNodeNameInDict]

            if placeNode.shouldRunMapper:
                setSubNodesShouldRunMapper(placeNode, sceneTopology[2], startIdx=1, flag=False)
            else:
                setSubNodesShouldRunMapper(placeNode, sceneTopology[2], startIdx=0, flag=False)

    # set all ShouldRunMapper of seqNodes as False, unless it is the only sub node of its parent
    for key in sceneTopology[3].keys():
        seqNode = sceneTopology[3][key]
        parentNodeNameInDict = getParentNodeNameInDict(seqNode.inheritNames, seqNode.type, seqNode.parent)
        parentNode = sceneTopology[2][parentNodeNameInDict]

        if parentNode.shouldRunMapper and (len(parentNode.subNodes) == 1):
            seqNode.shouldRunMapper = True
        else:
            seqNode.shouldRunMapper = False

    return

def setAllNodesShouldRunMapper(serialTopology, shouldRunMapper=True):
    for key, node in serialTopology.items():
        node.shouldRunMapper = shouldRunMapper

    return

def setPlaceNodesShouldRunMapper(sceneTopology, shouldRunMapper=True):
    # set default
    setAllNodesShouldRunMapper(getSerializedMapNodes(sceneTopology), True)

    # scene and palce
    for key in sceneTopology[0].keys():
        sceneNode = sceneTopology[0][key]
        setSubNodesShouldRunMapper(sceneNode, sceneTopology[1], startIdx=0, flag=shouldRunMapper)

        for i in range(len(sceneNode.mergeOrders)):
            placeNodeNameInDict = getSubNodeNameInDict(sceneNode.inheritNames, sceneNode.type,
                                                     sceneNode.mergeOrders[i])
            placeNode = sceneTopology[1][placeNodeNameInDict]

            if placeNode.shouldRunMapper:
                setSubNodesShouldRunMapper(placeNode, sceneTopology[2], startIdx=1, flag=False)
            else:
                setSubNodesShouldRunMapper(placeNode, sceneTopology[2], startIdx=0, flag=False)

    # set all ShouldRunMapper of seqNodes as False, unless it is the only sub node of its parent
    for key in sceneTopology[3].keys():
        seqNode = sceneTopology[3][key]
        parentNodeNameInDict = getParentNodeNameInDict(seqNode.inheritNames, seqNode.type, seqNode.parent)
        parentNode = sceneTopology[2][parentNodeNameInDict]

        if parentNode.shouldRunMapper and (len(parentNode.subNodes) == 1):
            seqNode.shouldRunMapper = True
        else:
            seqNode.shouldRunMapper = False

    return

def getMergeOrdersOfSubNodes(serialTopology):
    for key in serialTopology.keys():
        node = serialTopology[key]
        node.mergeOrders = getMergeOrders(node.subNodes, node.edges, interconnected=node.interconnected).copy()

    return

def setSeqCheckRunMapperFlags(seqNameList, sceneTopology):
    serialTopology = getSerializedMapNodes(sceneTopology)
    runMapperFlags = {}

    for seq in seqNameList:
        seqNode = serialTopology[seq]
        parentNodeNameInDict = getParentNodeNameInDict(seqNode.inheritNames, seqNode.type, seqNode.parent)
        parentNode = serialTopology[parentNodeNameInDict]

        if len(parentNode.subNodes) == 1:
            runMapperFlags[seq] = True
        else:
            runMapperFlags[seq] = False

    return runMapperFlags

def getRunMapperFlags(namesList, serialTopology):
    flags = {}
    for name in namesList:
        flags[name] = serialTopology[name].shouldRunMapper

    return flags

def readStrList(filePath):
    if not os.path.isfile(filePath):
        raise Exception('file: %s does not exist.' % filePath)

    lines = []
    with open(filePath) as fp:
        lines = fp.readlines()

    validStrList = []
    for line in lines:
        line = line.strip()
        if (len(line) > 1) and (not (line[0]=='#')):
            validStrList.append(line)

    return validStrList

def updateMergeOrders(updateFlags, serialTopology, dbDirPath):
    for key, importMergeOrders in updateFlags.items():
        if (key in serialTopology.keys()) and importMergeOrders:
            node = serialTopology[key]
            node.mergeOrders = readStrList(os.path.join(dbDirPath, key + '_mergeOrders.txt'))
            assert len(node.mergeOrders) == len(node.subNodes)

    return

def updateSceneNodeMergeOrders(supplePlaces, sceneNodeInfo, serialTopology):
    sceneNodeNameInDict = getNodeNameInDict(sceneNodeInfo['inheritNames'], sceneNodeInfo['type'])
    sceneNode = serialTopology[sceneNodeNameInDict]

    if ('importMergeOrders' in sceneNodeInfo.keys()) and sceneNodeInfo['importMergeOrders']:
        print('use imported merge orders.')
    else:
        sceneNode.mergeOrders = []
        for subNode in sceneNode.subNodes:
            if not (subNode in supplePlaces):
                sceneNode.mergeOrders.append(subNode)
        
        sceneNode.mergeOrders += supplePlaces
    
    mergeOrdersUpdated = sceneNode.mergeOrders.copy()

    if (not ('startMergeId' in sceneNodeInfo.keys())) or (sceneNodeInfo['startMergeId'] < 1):
        sceneNodeInfo['startMergeId'] = len(sceneNode.mergeOrders) - len(supplePlaces)
    
    return mergeOrdersUpdated

def getBaseReconTasks(sceneTopology):
    # sequences
    seqToReconList = getAllSeqListInRouteNodes(sceneTopology[2])

    # routes
    routeToReconList = getAllRouteListInRouteNodes(sceneTopology[2])

    # places
    placeToReconList = getAllNodesInfoInSerialTopology(sceneTopology[1])

    # scenes
    sceneToReconList = getAllNodesInfoInSerialTopology(sceneTopology[0])

    return seqToReconList, routeToReconList, placeToReconList, sceneToReconList

def getBaseReconTasksMapSupple(sceneTopology, supplePlaces):
    seqList, routeList, placeList, sceneList = getBaseReconTasks(sceneTopology)

    # scenes
    sceneToReconList = sceneList.copy()

    # places
    placeToReconList = []
    for placeInfo in placeList:
        if placeInfo['inheritNames'][-1] in supplePlaces:
            placeToReconList.append(placeInfo)

    # routes
    routeToReconList = []
    routeNamesInDict = []
    for routeInfo in routeList:
        if routeInfo['inheritNames'][-2] in supplePlaces:
            routeToReconList.append(routeInfo)
            routeNamesInDict.append(getNodeNameInDict(routeInfo['inheritNames'], routeInfo['type']))

    # sequence
    seqToReconList = getAllSeqListInRouteNodes(sceneTopology[2], routeNamesInDict)
 
    return seqToReconList, routeToReconList, placeToReconList, sceneToReconList

def str2bool(strValue):
    return strValue.lower() == "true"

def getBaseReconTasksSpecified(baseReconConfigFile, sceneTopology, dbDirPath):
    serialTopology = getSerializedMapNodes(sceneTopology)

    with open(baseReconConfigFile, 'r', encoding='UTF-8') as f:
        jsonValues = json.load(f)

    nodeConfigs = jsonValues["nodeConfig"]

    # update mergeOrders and shouldRunMapper
    updateFlags = {}
    for config in nodeConfigs:
        if config['nodeNameInDict'] in updateFlags.keys():
            raise Exception('repeated nodeNameInDict, please check if there are multiple infoDict about ' + config['nodeNameInDict'] + ' in baseReconConfig.json')
        else:
            if 'importMergeOrders' in config.keys():
                updateFlags[config['nodeNameInDict']] = config['importMergeOrders']
            else:
                updateFlags[config['nodeNameInDict']] = False

    updateMergeOrders(updateFlags, serialTopology, dbDirPath)
    setChildNodesShouldRunMapper(sceneTopology)

    # update other running config and get node info
    seqToReconList = []
    routeToReconList = []
    placeToReconList = []
    sceneToReconList = []

    for config in nodeConfigs:
        nodeNameInDict = config['nodeNameInDict']
        
        if not (nodeNameInDict in serialTopology.keys()):
            raise Exception(nodeNameInDict + ' is not in serialTopology.keys(), please check config file.')
        
        node = serialTopology[nodeNameInDict]
        if ('startMergeId' in config.keys()) and (config['startMergeId']>0):
            node.startMergeId = config['startMergeId']
        
        if 'importMergeOrders' in config.keys():
            node.importMergeOrders = config['importMergeOrders']

        if 'runBaseRecon' in config.keys():
            node.runBaseRecon = config['runBaseRecon']

        if 'sfmConfig' in config.keys():
            if 'runDBClean' in config['sfmConfig'].keys():
                node.runDBClean = config['sfmConfig']['runDBClean']

            if 'runFeatureExtractor' in config['sfmConfig'].keys():
                node.runFeatureExtractor = config['sfmConfig']['runFeatureExtractor']

            if 'runMatcher' in config['sfmConfig'].keys():
                node.runMatcher = config['sfmConfig']['runMatcher']

            if 'runMapper' in config['sfmConfig'].keys():
                node.runMapper = config['sfmConfig']['runMapper']

        if 'runScaleRec' in config.keys():
            node.runScaleRec = config['runScaleRec']

        if 'runGravityRec' in config.keys():
            node.runGravityRec = config['runGravityRec']

        if 'runCharucoDelete' in config.keys():
            node.runCharucoDelete = config['runCharucoDelete']

        nodeType, nodeInfo = getNodeInfo(nodeNameInDict, serialTopology)
        
        if ('startMergeId' in config.keys()) and (config['startMergeId']>0):
            nodeInfo['startMergeId'] = node.startMergeId
        
        if ('importMergeOrders' in config.keys()) and config['importMergeOrders']:
            nodeInfo['importMergeOrders'] = node.importMergeOrders

        if nodeType == 'sequence':
            seqToReconList.append(nodeInfo)
        elif nodeType == 'route':
            routeToReconList.append(nodeInfo)
        elif nodeType == 'place':
            placeToReconList.append(nodeInfo)
        elif nodeType == 'scene':
            sceneToReconList.append(nodeInfo)
        else:
            print('unsupported node type of %s for now' % nodeNameInDict)

    return seqToReconList, routeToReconList, placeToReconList, sceneToReconList

def getBaseReconTasksSpecified_duplicated(strLines, sceneTopology, dbDirPath, seperator=' '):
    serialTopology = getSerializedMapNodes(sceneTopology)

    validLines = []
    for line in strLines:
        line = line.strip()

        if len(line)<=0 or line[0]=='#':
            continue

        validLines.append(line)

    # update mergeOrders and shouldRunMapper
    updateFlags = {}
    for line in validLines:
        lineParts = line.split(seperator)
        if len(lineParts) == 3:
            updateFlags[lineParts[0]] = str2bool(lineParts[2])

    updateMergeOrders(updateFlags, serialTopology, dbDirPath)
    setChildNodesShouldRunMapper(sceneTopology)

    seqToReconList = []
    routeToReconList = []
    placeToReconList = []
    sceneToReconList = []

    for line in validLines:
        lineParts = line.split(seperator)

        if len(lineParts) <= 3:
            nodeType, nodeInfo = getNodeInfo(lineParts[0], serialTopology)

            if len(lineParts) >= 2:
                startMergeId = int(lineParts[1])
                assertStr = 'the second value in line is startMergeId, and supposed to be >= 1'
                assert startMergeId >= 1, assertStr
                nodeInfo['startMergeId'] = startMergeId

            if len(lineParts) == 3:
                importMergeOrders = str2bool(lineParts[2])
                nodeInfo['importMergeOrders'] = importMergeOrders

            if nodeType == 'sequence':
                seqToReconList.append(nodeInfo)
            elif nodeType == 'route':
                routeToReconList.append(nodeInfo)
            elif nodeType == 'place':
                placeToReconList.append(nodeInfo)
            elif nodeType == 'scene':
                sceneToReconList.append(nodeInfo)
            else:
                print('unsupported node type of %s for now' % lineParts[0])
        else:
            print('multiple values in line: %s\n' % line)

    return seqToReconList, routeToReconList, placeToReconList, sceneToReconList

def getSerializedMapNodes(sceneTopology):
    nodesDict = {}

    for i in range(len(sceneTopology)):
        nodesDict.update(sceneTopology[i])

    return nodesDict

def getHierarchicalMapNodes(serialTopology):
    sceneTopology = []
    sceneTopology.append({})  # scenes
    sceneTopology.append({})  # palces
    sceneTopology.append({})  # routes
    sceneTopology.append({})  # sequences

    for key, node in serialTopology.items():
        if node.type == 'scene':
            sceneTopology[0][key] = node
        elif node.type == 'place':
            sceneTopology[1][key] = node
        elif node.type == 'route':
            sceneTopology[2][key] = node
        elif node.type == 'sequence':
            sceneTopology[3][key] = node
        else:
            raise Exception('unknown node type: ' + node.type )

    return sceneTopology

def getNodeNameInDict(inheritNames, type):
    nameInDict = inheritNames[-1]

    if type == "route":
        nameInDict = inheritNames[-2] + '_' + nameInDict

    return nameInDict

def getSubNodeNameInDict(inheritNames, type, subNodeName):
    nameInDict = subNodeName

    if type == "place":
        nameInDict = inheritNames[-1] + '_' + nameInDict

    return nameInDict

def getParentNodeNameInDict(inheritNames, type, parentNodeName):
    nameInDict = parentNodeName

    if type == "sequence":
        nameInDict = inheritNames[-3] + '_' + nameInDict

    return nameInDict

def getSubNodeNamesInDict(inheritNames, type, subNodeNames):
    namesInDict = subNodeNames.copy()

    if type == "place":
        for i in range(len(namesInDict)):
            namesInDict[i] = inheritNames[-1] + '_' + namesInDict[i]

    return namesInDict

def getAssocEgedsByOneNodeName(edges, nodeName):
    nodeIdxInEdges = []
    asscoEdges = []
    edgeIdxs = []

    for i in range(len(edges)):
        edge = edges[i]
        if nodeName in edge:
            nodeIdxInEdges.append(edge.index(nodeName))
            asscoEdges.append(edge)
            edgeIdxs.append(i)

    return asscoEdges, edgeIdxs, nodeIdxInEdges

def getNodeDict(node, idx, serialTopology=None, parentNode=None):
    nodeDict = {}

    nodeDict['name'] = node.name
    nodeDict['idx'] = idx
    nodeDict['type'] = node.type

    if not (node.parent == ''):
        nodeDict['parent'] = node.parent

    nodeDict['members'] = {}
    nodeDict['members']['indices'] = []
    subNodesDict = {}
    for i in range(len(node.subNodes)):
        subNodeIdx = i + 1
        subNodeName = node.subNodes[i]
        subNodesDict[subNodeName] = subNodeIdx
        if node.type == 'place':
            subNodeNameInDict = getSubNodeNameInDict(node.inheritNames, node.type, subNodeName)
            subNode = serialTopology[subNodeNameInDict]
            nodeDict['members']['indices'].append([subNodeIdx, subNodeName, subNode.cameraModel, subNode.focalLenFactor, subNode.intervalForMatch])
        else:
            nodeDict['members']['indices'].append([subNodeIdx, subNodeName])

    nodeDict['members']['edges'] = []
    for edge in node.edges:
        nodeDict['members']['edges'].append([subNodesDict[edge[0]], subNodesDict[edge[1]]])

    if node.type == 'place':
        nodeDict['members']['charuco_ids'] = []
        nodeDict['members']['charuco_pairs'] = []
        for charucoRoute in node.charucoRoutes:
            nodeDict['members']['charuco_ids'].append(subNodesDict[charucoRoute])

            asscoEdges, _, nodeIdxInEdges = getAssocEgedsByOneNodeName(node.edges, charucoRoute)

            for i in range(len(asscoEdges)):
                assocRouteName = asscoEdges[i][(nodeIdxInEdges[i]+1) % 2]
                nodeDict['members']['charuco_pairs'].append([subNodesDict[charucoRoute],
                                                             subNodesDict[assocRouteName]])

    if not (parentNode == None):
        nodeDict['neighbors'] = []
        asscoParentEdges, edgeIdxInParentEdges, nodeIdxInParentEdges = getAssocEgedsByOneNodeName(parentNode.edges, node.name)

        for i in range(len(asscoParentEdges)):
            neighborDict = {}
            neighborDict['name'] = asscoParentEdges[i][(nodeIdxInParentEdges[i]+1) % 2]
            neighborDict['edges'] = []
            for hook in parentNode.edgeHooks[edgeIdxInParentEdges[i]]:
                neighborDict['edges'].append([hook[nodeIdxInParentEdges[i]], hook[(nodeIdxInParentEdges[i]+1) % 2]])

            nodeDict['neighbors'].append(neighborDict)

    return nodeDict

def writeSceneTolopogy(sceneTopology, topologyFile):
    topologyDict = {}
    topologyDict['Items'] = []

    # get node dicts
    for i in range(2):
        nodeIdx = 0
        for name, node in sceneTopology[i].items():
            nodeIdx += 1
            parentNodeName = node.parent
            parentNode = None
            if i > 0:
                parentNode = sceneTopology[i-1][parentNodeName]

            nodeDict = getNodeDict(node, nodeIdx, serialTopology=sceneTopology[2], parentNode=parentNode)

            topologyDict['Items'].append(nodeDict)

    with open(topologyFile, 'w') as fp:
        json.dump(topologyDict, fp, indent=4)

    return

def getEdgeHooksInName(sceneTopology):
    for key in sceneTopology[0].keys():
        sceneNode = sceneTopology[0][key]
        sceneNode.edgeHooksInName = []

        for i in range(len(sceneNode.edges)):
            edge = sceneNode.edges[i]
            edgeHooksInName = []
            for hook in sceneNode.edgeHooks[i]:
                hookInName = []
                for j in range(2):
                    placeName = edge[j]
                    routeIdx = hook[j]

                    placeNode = sceneTopology[1][placeName]
                    routeName = placeNode.subNodes[routeIdx-1]
                    hookInName.append(routeName)
                edgeHooksInName.append(hookInName)

            sceneTopology[0][key].edgeHooksInName.append(edgeHooksInName)

    return

def updateEdgeHooksByName(sceneTopology):
    for key in sceneTopology[0].keys():
        sceneNode = sceneTopology[0][key]

        for i in range(len(sceneNode.edges)):
            edge = sceneNode.edges[i]
            sceneNode.edgeHooks[i] = []

            for hookInName in sceneNode.edgeHooksInName[i]:
                hook = []
                for j in range(2):
                    placeName = edge[j]
                    routeName = hookInName[j]
                    routeIdx = sceneTopology[1][placeName].subNodes.index(routeName) + 1
                    hook.append(routeIdx)

                sceneNode.edgeHooks[i].append(hook)

    return

def deleteRouteNodes(serialTopology, routeInfoToDelete):
    for routeInfo in routeInfoToDelete:
        routeNameInDict = getNodeNameInDict(routeInfo['inheritNames'], routeInfo['type'])
        routeNode = serialTopology[routeNameInDict]
        seqNodesList = routeNode.subNodes

        # delete associated sequence nodes
        for seqName in seqNodesList:
            serialTopology.pop(seqName)

        # delete route node
        serialTopology.pop(routeNameInDict)

        # delete associated subNode and charucoRoute in parent Place
        parent = routeInfo['inheritNames'][-2]

        if routeInfo['inheritNames'][-1] in serialTopology[parent].subNodes:
            serialTopology[parent].subNodes.remove(routeInfo['inheritNames'][-1])

        if routeInfo['inheritNames'][-1] in serialTopology[parent].charucoRoutes:
            serialTopology[parent].charucoRoutes.remove(routeInfo['inheritNames'][-1])

    return

def deleteRoutePairs(serialTopology, routePairInfoToDelete):
    for routePair in routePairInfoToDelete:
        routeInfo1 = routePair[0]
        place1 = routeInfo1['inheritNames'][-2]
        route1 = routeInfo1['inheritNames'][-1]

        routeInfo2 = routePair[1]
        place2 = routeInfo2['inheritNames'][-2]
        route2 = routeInfo2['inheritNames'][-1]

        if place1 == place2:
            placeNode = serialTopology[place1]
            matchedEdgeIdx, _, _ = getMatchedEdge([route1, route2], placeNode.edges)
            if matchedEdgeIdx >= 0:
                placeNode.edges.pop(matchedEdgeIdx)
            else:
                raise Exception('edge to be deleted is not in placeNode.edges')

        else:
            sceneNode = serialTopology[routeInfo1['inheritNames'][-3]]
            matchedEdgeIdx, tarNode1IdxInMatchedEdge, tarNode2IdxInMatchedEdge = getMatchedEdge([place1, place2], sceneNode.edges)
            if matchedEdgeIdx >= 0:
                hookInName = ['', '']
                hookInName[tarNode1IdxInMatchedEdge] = route1
                hookInName[tarNode2IdxInMatchedEdge] = route2

                if hookInName in sceneNode.edgeHooksInName[matchedEdgeIdx]:
                    sceneNode.edgeHooksInName[matchedEdgeIdx].remove(hookInName)
                else:
                    raise Exception('hookInName to be deleted is not in sceneNode.edgeHooksInName')

            else:
                raise Exception('edge to be deleted is not in sceneNode.edges')

    return

def cleanSceneTopology(routeStatusDict, routePairStatusDict, routeInfoList, routePairInfoList, sceneTopology):
    # generate edgeHooksInName according to edgeHooks
    getEdgeHooksInName(sceneTopology)
    serialTopology = getSerializedMapNodes(sceneTopology)

    # get routeInfoToDelete and routePairInfoToDelete
    routeInfoToDelete = []
    routeDictNameToDelete = []
    for routeInfo in routeInfoList:
        routeNameInDict = getNodeNameInDict(routeInfo['inheritNames'], routeInfo['type'])
        if (not (routeNameInDict in routeStatusDict.keys())) or (not (routeStatusDict[routeNameInDict]=='success')):
            routeInfoToDelete.append(routeInfo)
            routeDictNameToDelete.append(routeNameInDict)

    routePairInfoToDelete = []
    for routePairInfo in routePairInfoList:
        routeNameInDict1 = getNodeNameInDict(routePairInfo[0]['inheritNames'], routePairInfo[0]['type'])
        routeNameInDict2 = getNodeNameInDict(routePairInfo[1]['inheritNames'], routePairInfo[1]['type'])
        routePairName = routeNameInDict1 + '_' + routeNameInDict2

        if (not (routePairName in routePairStatusDict.keys()))\
                or (not (routePairStatusDict[routePairName] == 'success'))\
                or (routeNameInDict1 in routeDictNameToDelete) \
                or (routeNameInDict2 in routeDictNameToDelete):
            routePairInfoToDelete.append(routePairInfo)

    # delete route sceneTopology, along with associated subnode, charucoRoute and sequence
    deleteRouteNodes(serialTopology, routeInfoToDelete)

    # delete route pair (i.e., place edge or scene hook ) in sceneTopology
    deleteRoutePairs(serialTopology, routePairInfoToDelete)

    # serialTopology to hierarchicalTolopology
    sceneTopologyUpdated = getHierarchicalMapNodes(serialTopology)

    # generate edgeHooks according to cleaned edgeHooksInName
    updateEdgeHooksByName(sceneTopologyUpdated)

    # update charucoSeqs for each place and scene
    getCharucoSeqs(sceneTopologyUpdated)

    return sceneTopologyUpdated

if __name__ == "__main__":
    projPath = '/media/lzx/文档/lzx-data/ProjParkC'
    projName = 'ProjParkC'
    batchName = 'P1build'

    sceneTopology = getSceneTopology(projPath, projName, batchName)
    topologyFile = os.path.join(projPath, "scene_graph_checked.json")
    #writeSceneTolopogy(sceneTopology, topologyFile)

    sceneTopology2 = getSceneTopology(projPath, projName, batchName, sceneGraphName="scene_graph_checked")

    #seqToCheckList, routeToCheckList, routePairToCheckList = getTopologyCheckTasks(sceneTopology)

    print("done.\n")

