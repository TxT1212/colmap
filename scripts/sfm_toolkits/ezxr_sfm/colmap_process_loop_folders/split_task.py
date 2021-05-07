# coding: utf-8
import os
import sys
import argparse

sys.path.append('../')

from colmap_process_loop_folders.tree_topology import getSceneTopology, getTopologyCheckTasks, \
    getBaseReconTasks, getOnlyNamesFromNodesInfo

def saveCheckTaskList(allCheckList, taskPath, sceneGraphName):
    checkTaskPath = os.path.join(taskPath, 'tpCheck', sceneGraphName)
    if not os.path.isdir(checkTaskPath):
        os.makedirs(checkTaskPath)

    checkListFile = os.path.join(checkTaskPath, 'checklist.txt')

    with open(checkListFile, 'w') as fp:
        for checkList in allCheckList:
            for item in checkList:
                if type(item) is str:
                    fp.write(item + '\n')
                elif type(item) is list:
                    fp.write(item[0] + ' ' + item[1] + '\n')
                else:
                    raise Exception('unsupported item type')

    return

def saveReconTaskList(allReconList, taskPath, sceneGraphName):
    reconTaskPath = os.path.join(taskPath, 'baseRecon', sceneGraphName)
    if not os.path.isdir(reconTaskPath):
        os.makedirs(reconTaskPath)

    reconListFile = os.path.join(reconTaskPath, 'reconlist.txt')

    with open(reconListFile, 'w') as fp:
        for reconList in allReconList:
            for item in reconList:
                if type(item) is str:
                    fp.write(item + '\n')
                else:
                    raise Exception('unsupported item type')

    return

def run_check_task_split(args):
    configPath = os.path.join(args.projPath, args.configDir)

    print('start scene graph parsing...\n')
    if args.validSeqsFromJson:
        sceneTopology = getSceneTopology(args.projPath, configPath=configPath, sceneGraphName=args.sceneGraph,
                                         validSeqsFromJson=args.validSeqsFromJson,
                                         suppleGraphName=args.supplementalGraph)
    else:
        sceneTopology = getSceneTopology(args.projPath, projName=args.projName, batchName=args.batchName,
                                         sceneGraphName=args.sceneGraph,
                                         suppleGraphName=args.supplementalGraph)

    taskPath = os.path.join(args.projPath, args.taskDir)
    seqList, routeList, routePairList = getTopologyCheckTasks(sceneTopology)
    only_name_list = getOnlyNamesFromNodesInfo([seqList, routeList, routePairList])
    saveCheckTaskList(only_name_list, taskPath, args.sceneGraph)

    return

def run_recon_task_split(args):
    configPath = os.path.join(args.projPath, args.configDir)

    print('start scene graph parsing...\n')
    if args.validSeqsFromJson:
        sceneTopology = getSceneTopology(args.projPath, configPath=configPath, sceneGraphName=args.sceneGraph,
                                            validSeqsFromJson=args.validSeqsFromJson,
                                            suppleGraphName=args.supplementalGraph)
    else:
        sceneTopology = getSceneTopology(args.projPath, projName=args.projName, batchName=args.batchName,
                                            sceneGraphName=args.sceneGraph,
                                            suppleGraphName=args.supplementalGraph)

    seqList, routeList, placeList, sceneList = getBaseReconTasks(sceneTopology)
    reconListOnlyNames = getOnlyNamesFromNodesInfo([seqList, routeList, placeList, sceneList])
    saveReconTaskList(reconListOnlyNames, os.path.join(args.projPath, args.taskDir), args.sceneGraph)

    return

def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--taskType', required=True, choices=['tpCheck', 'baseRecon'])
    parser.add_argument('--projPath', required=True)
    parser.add_argument('--projName', type=str, default=None, help='如果使用了--validSeqsFromJson，则必须指定该参数')
    parser.add_argument('--batchName', type=str, default=None, help='如果使用了--validSeqsFromJson，则必须指定该参数')

    parser.add_argument('--configDir', default='config')
    parser.add_argument('--taskDir', default='tasks')
    parser.add_argument('--validSeqsFromJson', action='store_false')
    parser.add_argument('--sceneGraph', type=str, default='scene_graph')
    parser.add_argument('--supplementalGraph', type=str, default='')

    args = parser.parse_args()

    return args


def main():
    args = parseArgs()
    if args.taskType == 'tpCheck':
        run_check_task_split(args)
    else:
        run_recon_task_split(args)

    return

if __name__ == "__main__":
    main()