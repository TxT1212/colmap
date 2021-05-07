import os
import argparse

def writePlyVC(filename, verts, colors):
    fp = open(filename, 'w')
    numVerts = len(verts)

    fp.write('ply\nformat ascii 1.0\n')

    fp.write('element vertex %d\n' % numVerts)
    fp.write('property float x\nproperty float y\nproperty float z\n'
             'property uchar red\nproperty uchar green\nproperty uchar blue\n')

    fp.write('end_header\n')

    for i in range(numVerts):
        fp.write('%f %f %f %u %u %u\n' %
                 (verts[i][0], verts[i][1], verts[i][2],
                  colors[i][0], colors[i][1], colors[i][2]))
    fp.close()

def geos2ply(geosFile, color=[255, 0, 0]):
    lines = []
    with open(geosFile) as fp:
        lines = fp.readlines()

    verts = []
    colors = []
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue

        lineParts = line.split(' ')
        verts.append([float(lineParts[1]), float(lineParts[2]), float(lineParts[3])])
        colors.append(color.copy())

    fileDirPath, fileName = os.path.split(geosFile)
    name, ext = os.path.splitext(fileName)
    plyFile = os.path.join(fileDirPath, name+'.ply')

    writePlyVC(plyFile, verts, colors)

    return

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--geosFile', required=True, type=str, help='full path of geos.txt')
    parser.add_argument('--color', type=int, nargs='+', default=[255, 0, 0], help='r g b color of geo points')

    args = parser.parse_args()

    return args

def main():
    args = parseArgs()
    geos2ply(args.geosFile, color=args.color)

if __name__ == '__main__':
    main()
