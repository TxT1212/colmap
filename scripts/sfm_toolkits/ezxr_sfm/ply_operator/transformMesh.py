import os
import transform as trans
import numpy as np
from glob import  glob
import argparse
def transformObj(inputFile, R, T, outSuffix='-aligned'):
    parDir, filename = os.path.split(inputFile)
    name, ext =  os.path.splitext(filename)
    outFile = os.path.join(parDir, name + outSuffix + ext)

    lines = []
    with open(inputFile) as fInput:
        lines = fInput.readlines()

    print('transform v and vn...\n')

    for i in range(len(lines)):
        line = lines[i]
        if len(line) > 0 and line[0:2] == 'v ':
            line_parts = line.split(' ')
            vert = np.array([float(line_parts[1]), float(line_parts[2]), float(line_parts[3])])
            outVert = R.dot(vert) + T

            outLine = 'v %f %f %f\n' % (outVert[0], outVert[1], outVert[2])
            lines[i] = outLine
        elif len(line) > 0 and line[0:2] == 'vn':
            line_parts = line.split(' ')
            norm = np.array([float(line_parts[1]), float(line_parts[2]), float(line_parts[3])])
            outNorm = R.dot(norm)

            outLine = 'vn %f %f %f\n' % (outNorm[0], outNorm[1], outNorm[2])
            lines[i] = outLine
        else:
            continue

    print('write back to file...\n')
    with open(outFile, 'w') as fOutput:
        for line in lines:
            fOutput.write('%s' % line)

    print('done.\n')

def trasformPlyVC(inputFile, R, T, outSuffix='-aligned', outputFile=None):
    parDir, filename = os.path.split(inputFile)
    name, ext = os.path.splitext(filename)
    outFile = os.path.join(parDir, name + outSuffix + ext)

    if not(outputFile==None): outFile = outputFile

    lines = []
    with open(inputFile) as fInput:
        lines = fInput.readlines()

    print('transform v\n')

    # find numVerts and idxEndHeader
    numVerts = 0
    idxEndHeader = 0
    for i in range(len(lines)):
        line = lines[i].strip(' ')

        if len(line) >= 14 and line[0:14] == 'element vertex':
            line_parts = line.split(' ')
            numVerts = int(line_parts[2])
        elif len(line) >= 10 and line[0:10] == 'end_header':
            idxEndHeader = i
            break
        else:
            continue

    for i in range(numVerts):
        idxInLines = idxEndHeader + 1 + i
        line = lines[idxInLines].strip(' \n')
        line_parts = line.split(' ')
        vert = np.array([float(line_parts[0]), float(line_parts[1]), float(line_parts[2])])
        outVert = R.dot(vert) + T

        if len(line_parts)==6:
            color = np.array([int(line_parts[3]), int(line_parts[4]), int(line_parts[5])])
            outLine = '%f %f %f %u %u %u\n' % (outVert[0], outVert[1], outVert[2],
                                               color[0], color[1], color[2])
        elif len(line_parts)==7:
            color = np.array([int(line_parts[3]), int(line_parts[4]), int(line_parts[5]), int(line_parts[6])])
            outLine = '%f %f %f %u %u %u %u\n' % (outVert[0], outVert[1], outVert[2],
                                                  color[0], color[1], color[2], color[3])
        else:
            print('wrong ply file.\n')
            return

        lines[idxInLines] = outLine

    print('write back to file...\n')
    with open(outFile, 'w') as fOutput:
        for line in lines:
            fOutput.write('%s' % line)

    print('done.\n')

def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--plyFile', type=str, required=True, help="暂时只支持ascii格式的ply文件，且无法向信息")
    parser.add_argument('--rtFile', type=str, required=True, help="txt文件，第一行是行主序的旋转矩阵，第二行是平移向量，数字间以空格隔开")
    parser.add_argument('--outSuffix', default='_aligned')
    parser.add_argument('--outputFile', default=None, type=str)

    args = parser.parse_args()

    return args

def main():
    args = parseArgs()
    R, T = trans.readRT(args.rtFile)
    trasformPlyVC(args.plyFile, R, T, outSuffix=args.outSuffix, outputFile=args.outputFile)

if __name__=='__main__':
    main()

    

