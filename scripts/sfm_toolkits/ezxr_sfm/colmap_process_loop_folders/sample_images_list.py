# coding: utf-8
import os
import sys
import argparse

sys.path.append('../')

def sample_file_lines(srcFile, dstFile, interval=1):
    print('sample lines in %s to %s with interval = %d' % (srcFile, dstFile, interval))

    lines = []
    with open(srcFile) as fp:
        lines = fp.readlines()

    with open(dstFile, 'w') as fp:
        for i in range(0, len(lines), interval):
            fp.write(lines[i])

    initNumLines = len(lines)
    sampledNumLines = int(initNumLines / interval)

    print('done.')

    return initNumLines, sampledNumLines

def sample_image_list(args):
    srcDir = args.srcDir
    dstDir = args.dstDir
    if (srcDir[-1]=='/') or (srcDir[-1]=='\\'):
        srcDir = srcDir[0:-1]
    
    parentDir, dirName = os.path.split(srcDir)
    if dstDir==None:
        dstDir = os.path.join(parentDir, dirName + '_sampled_' + str(args.interval))
    
    if not os.path.isdir(dstDir):
        os.mkdir(dstDir)
    
    fileList = os.listdir(srcDir)

    initNumLinesSum = 0
    sampledNumLinesSum = 0

    for filename in fileList:
        name, ext = os.path.splitext(filename)
        srcFilePath = os.path.join(srcDir, filename)

        if os.path.isfile(srcFilePath):
            dstFilePath = os.path.join(dstDir, name + args.suffix + ext)
            initNumLines, sampledNumLines = sample_file_lines(srcFilePath, dstFilePath, interval=args.interval)
            initNumLinesSum += initNumLines
            sampledNumLinesSum += sampledNumLines

    print(str(initNumLinesSum) + ' lines in original files,\n' + str(sampledNumLinesSum) + ' lines in sampled files.')

    return

def replace_last(source_string, replace_what, replace_with):
    head, _sep, tail = source_string.rpartition(replace_what)
    return head + replace_with + tail

def replace_suffix(srcDir, old_suffix, new_suffix):
    if (srcDir[-1]=='/') or (srcDir[-1]=='\\'):
        srcDir = srcDir[0:-1]
    
    parentDir, dirName = os.path.split(srcDir)
    dstDir = os.path.join(parentDir, dirName + '_sampled_1')
    
    if not os.path.isdir(dstDir):
        os.mkdir(dstDir)
    
    fileList = os.listdir(srcDir)

    initNumLinesSum = 0
    sampledNumLinesSum = 0

    for filename in fileList:
        name, ext = os.path.splitext(filename)
        srcFilePath = os.path.join(srcDir, filename)

        if os.path.isfile(srcFilePath):
            new_name = replace_last(name, old_suffix, new_suffix)
            dstFilePath = os.path.join(dstDir, new_name + ext)
            initNumLines, sampledNumLines = sample_file_lines(srcFilePath, dstFilePath, interval=1)
            initNumLinesSum += initNumLines
            sampledNumLinesSum += sampledNumLines

    print(str(initNumLinesSum) + ' lines in original files,\n' + str(sampledNumLinesSum) + ' lines in sampled files.')
    return

def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--srcDir', required=True, help='full path of dir containing imagelist files')
    parser.add_argument('--dstDir', default=None, type=str, help='ull path of dir to save sampled imagelist files')
    parser.add_argument('--interval', default='5', type=int)
    parser.add_argument('--suffix', default='', type = str)
    parser.add_argument('--replace_suffix', action='store_true')
    parser.add_argument('--old_suffix', default='', type = str)
    parser.add_argument('--new_suffix', default='', type = str)

    args = parser.parse_args()
    return args

def main():
    args = parseArgs()
    if args.replace_suffix:
        replace_suffix(args.srcDir, args.old_suffix, args.new_suffix)
    else:
        sample_image_list(args)

    return


if __name__ == '__main__':
    main()