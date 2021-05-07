# coding: utf-8
import os, sys
sys.path.append('../')
import argparse
import colmap_process_loop_folders.basic_colmap_operation as bco
from colmap_process.colmap_seq_sfm import run_custom_sfm

def readStrPairs(file):
    lines = []
    with open(file) as fp:
        lines = fp.readlines()

    outPairs = []
    for line in lines:
        line = line.strip()
        lineParts = line.split(' ')

        if len(lineParts) == 2:
            outPairs.append([lineParts[0], lineParts[1]])

    return outPairs

def readStrLines(file):
    lines = []
    with open(file) as fp:
        lines = fp.readlines()

    outLines = []
    for line in lines:
        line = line.strip()
        if len(line) > 0:
            outLines.append(line)

    return outLines

def run_scenes_joiner(args):
    imagesPath = os.path.join(args.projPath, args.imagesDir)
    dbPath = os.path.join(args.projPath, args.dbDir)
    modelPath = os.path.join(args.projPath, args.modelDir)

    # read jointSeqsList and jointSeqsMatches
    seqList = readStrLines(os.path.join(args.projPath, args.jointName + '_nodeList.txt'))
    seqMatches = readStrPairs(os.path.join(args.projPath, args.jointName + '_nodeMatches.txt'))

    # merge imageList, db
    imageListFiles = []
    dbFiles = []
    for seq in seqList:
        imageListFiles.append(os.path.join(imagesPath, seq+args.imageListSuffix+'.txt'))
        dbFiles.append(os.path.join(dbPath, seq+args.imageListSuffix+'.db'))

    jointImageListFile = os.path.join(imagesPath, args.jointName+args.imageListSuffix+'.txt')
    jointDBFile = os.path.join(dbPath, args.jointName + args.imageListSuffix + '.db')
    jointMatchListFile = os.path.join(dbPath, args.jointName + args.imageListSuffix + '_match.txt')
    jointModelPath = os.path.join(modelPath, args.jointName + args.imageListSuffix)


    bco.mergeImageListFile(imageListFiles, jointImageListFile)
    bco.mergeDBFile(dbFiles, jointDBFile, colmapPath=args.colmapPath)
    bco.createMatchList(imagesPath, seqMatches, jointMatchListFile, imageListSuffix=args.imageListSuffix)

    inputModelPath = bco.getOneValidModelPath([os.path.join(modelPath, seqList[0] + args.imageListSuffix)])
    run_custom_sfm(args.colmapPath, jointDBFile, imagesPath, jointImageListFile, jointMatchListFile, jointModelPath,
                   matcher_min_num_inliers=15, mapper_min_num_matches=50,
                   mapper_init_min_num_inliers=200,
                   mapper_abs_pose_min_num_inliers=60,
                   mapper_input_model_path=inputModelPath,
                   skip_feature_extractor=True,
                   apply_model_aligner=False,
                   mapper_fix_existing_images=1)

    return

def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--projPath', type=str, required=True)
    parser.add_argument('--jointName', required=True)

    parser.add_argument('--imagesDir', default='images')
    parser.add_argument('--masksDir', default=None, type=str)
    parser.add_argument('--dbDir', default='database')
    parser.add_argument('--modelDir', default='sparse')
    parser.add_argument('--taskDir', default='tasks')
    parser.add_argument('--imageListSuffix', default='', type=str)
    parser.add_argument('--configDir', default='config')

    parser.add_argument('--colmapPath', default="colmap")
    parser.add_argument('--optionalMapper', action='store_true')

    args = parser.parse_args()

    return args

def main():
    args = parseArgs()
    run_scenes_joiner(args)

if __name__ == '__main__':
    main()