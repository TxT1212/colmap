# coding: utf-8
import os
import numpy
import sys
from plyfile import PlyData, PlyElement

def main():
    if len(sys.argv) != 2:
        print(sys.argv[0], ' ply_file')
        return

    inputPlyFile = sys.argv[1]
    if os.path.isabs(inputPlyFile):
        fileprefix = os.path.splitext(inputPlyFile)[0]
    else:
        fileprefix = os.getcwd() + '/' + os.path.splitext(inputPlyFile)[0]
    
    outputPlyFile = fileprefix + '_unity.ply'
    
    plydata = PlyData.read(inputPlyFile)

    x = plydata['vertex']['x'].copy()
    y = plydata['vertex']['y'].copy()
    z = plydata['vertex']['z'].copy()
    nx = plydata['vertex']['nx'].copy()
    ny = plydata['vertex']['ny'].copy()
    nz = plydata['vertex']['nz'].copy()

    plydata['vertex']['x'] = x
    plydata['vertex']['y'] = z
    plydata['vertex']['z'] = y

    plydata['vertex']['nx'] = nx
    plydata['vertex']['ny'] = nz
    plydata['vertex']['nz'] = ny

    print("Write to ", outputPlyFile, "\nwith inverse y and z")
    PlyData.write(plydata, outputPlyFile)

if __name__ == '__main__':
    main()
