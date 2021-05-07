# coding: utf-8
import os, sys
sys.path.append('../')
import argparse
import shutil
from colmap_process.colmap_read_write_model import read_images_binary, read_cameras_binary
import cv2
import numpy as np

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def checkDirAndMake(dirName):
    if not os.path.isdir(dirName):
        os.mkdir(dirName)

    return

def createMVEDataDir(projPath, mveDataDir, modelName):
    mveDataPath = os.path.join(projPath, mveDataDir, modelName)
    if not os.path.isdir(mveDataPath):
        os.makedirs(mveDataPath)

    texturePath = os.path.join(mveDataPath, 'texture')
    if os.path.isdir(texturePath):
        shutil.rmtree(texturePath)

    os.mkdir(texturePath)

    viewsPath = os.path.join(mveDataPath, 'views')
    if os.path.isdir(viewsPath):
        shutil.rmtree(viewsPath)

    os.mkdir(viewsPath)

    gatheredViewsPath = os.path.join(mveDataPath, 'viewsGathered')
    if os.path.isdir(gatheredViewsPath):
        shutil.rmtree(gatheredViewsPath)

    os.mkdir(gatheredViewsPath)

    return mveDataPath, texturePath, viewsPath, gatheredViewsPath

def getCameraIntrin(camera):
    if camera.model == 'SIMPLE_PINHOLE':
        fx, cx, cy = camera.params
        fy = fx
    elif camera.model == 'PINHOLE':
        fx, fy, cx, cy = camera.params
    elif camera.model == 'SIMPLE_RADIAL':
        fx, cx, cy, k = camera.params
        fy = fx
    elif camera.model == 'RADIAL':
        fx, cx, cy, k1, k2 = camera.params
        fy = fx
    else:
        raise Exception('unsupported camera model for now: ' + camera.model)

    cameraIntrin = {'width': camera.width,
                    'height': camera.height,
                    'fx': fx,
                    'fy': fy,
                    'cx': cx,
                    'cy': cy}

    return cameraIntrin

def getImagesWithCameraInfo(sparseModelPath):
    colmapImagesFile = os.path.join(sparseModelPath, 'images.bin')
    colmapCamerasFile = os.path.join(sparseModelPath, 'cameras.bin')

    images = read_images_binary(colmapImagesFile)
    cameras = read_cameras_binary(colmapCamerasFile)

    nameList = []
    intrinList = []
    rList = []
    tList = []
    for imageID, image in images.items():
        nameList.append(image.name)

        intrinList.append(getCameraIntrin(cameras[image.camera_id]))

        rList.append(image.qvec2rotmat())
        tList.append(image.tvec)

    nameListSorted, intrinListSorted, rListSorted, tListSorted = zip(*sorted(zip(nameList, intrinList, rList, tList)))

    return nameListSorted, intrinListSorted, rListSorted, tListSorted

def writeIniFile(filename, cameraIntrin, rotmat, tvec, viewIdx):
    with open(filename,'w') as fp:
        fp.write('# MVE view meta data is stored in INI-file syntax.\n'
                 '# This file is generated, formatting will get lost.\n')

        fp.write('\n[camera]\n')
        # intrinsics
        if cameraIntrin['width'] < cameraIntrin['height']:
            focalLength = cameraIntrin['fy'] / cameraIntrin['height']
        else:
            focalLength = cameraIntrin['fx'] / cameraIntrin['width']

        pixelAspect = cameraIntrin['fy'] / cameraIntrin['fx']
        principalPointX = cameraIntrin['cx'] / cameraIntrin['width']
        principalPointY = cameraIntrin['cy'] / cameraIntrin['height']

        fp.write('focal_length = %f\n' % focalLength)
        fp.write('pixel_aspect = %f\n' % pixelAspect)
        fp.write('principal_point = %f %f\n' % (principalPointX, principalPointY))

        # extrinsics
        fp.write('rotation = %f %f %f %f %f %f %f %f %f\n' %
                 (rotmat[0, 0], rotmat[0, 1], rotmat[0, 2],
                  rotmat[1, 0], rotmat[1, 1], rotmat[1, 2],
                  rotmat[2, 0], rotmat[2, 1], rotmat[2, 2]))
        fp.write('translation = %f %f %f\n' % (tvec[0], tvec[1], tvec[2]))

        # view name and index
        fp.write('\n[view]\n')
        fp.write('id = %d\nname = %04d\n' % (viewIdx, viewIdx))

    return


def writeViewsData(imagesPath, viewsPath, nameList, intrinList, rList, tList, scale=1.0, interval=10, gatheredViewsPath=None):
    numImages = len(nameList)
    viewCnt = 0
    for i in range(0, numImages, interval):
        name = nameList[i]
        relativeDir, _ = os.path.split(name)
        _, ext = os.path.splitext(name)

        # each viewDir
        viewDir = os.path.join(viewsPath, 'view_%06d.mve' % viewCnt)
        checkDirAndMake(viewDir)

        # viewImage
        srcImage = os.path.join(imagesPath, name)
        dstImage = os.path.join(viewDir, 'out_RGB' + ext)

        if not (gatheredViewsPath==None):
            dstImageGathered = os.path.join(gatheredViewsPath, name)
            viewsGatheredSubPath = os.path.join(gatheredViewsPath, relativeDir)

        if not os.path.isdir(viewsGatheredSubPath):
            os.makedirs(viewsGatheredSubPath)

        if abs(scale - 1.0) < 1e-5:
            shutil.copy(srcImage, dstImage)

            if not (gatheredViewsPath==None):
                shutil.copy(srcImage, dstImageGathered)
        else:
            imageMat = cv2.imread(srcImage)
            imageMatResized = cv2.resize(imageMat, dsize=None, fx=scale, fy=scale,
                                    interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(dstImage, imageMatResized)

            if not (gatheredViewsPath==None):
                cv2.imwrite(dstImageGathered, imageMatResized)

        # ini file
        iniFile = os.path.join(viewDir, 'meta.ini')
        writeIniFile(iniFile, intrinList[i], rList[i], tList[i], viewCnt)

        viewCnt += 1

    print('%d views prapared.\n' % viewCnt)
    return

def run_mesh_texturing(args):
    # create file dirs of mveData
    mveDataPath, texturePath, viewsPath, gatheredViewsPath = createMVEDataDir(args.projPath, args.mveDataDir, args.modelName)

    # get aligned images and camera pose
    densePath = os.path.join(args.projPath, args.denseDir, args.modelName)
    sparseModelPath = os.path.join(densePath, 'sparse')
    nameList, intrinList, rList, tList = getImagesWithCameraInfo(sparseModelPath)

    # generate views data
    imagesPath = os.path.join(args.projPath, args.imagesDir)
    writeViewsData(imagesPath, viewsPath, nameList, intrinList, rList, tList,
     scale=args.scale, interval=args.interval, gatheredViewsPath=gatheredViewsPath)

    # generate mvs-texturing command
    plyModelFile = os.path.join(densePath, 'ply', args.plyModelName + '.ply')

    appDirPath, appName = os.path.split(args.appPath)
    command = 'cd %s\n ./%s %s::out_RGB %s %s/%s --keep_unseen_faces' % \
              (appDirPath, appName, mveDataPath, plyModelFile, texturePath, args.modelName)

    if (args.persistOneAtlas):
        command += ' --persist_one_atlas'

    bashFile = os.path.join(mveDataPath, 'run_texrecon.bash')
    with open(bashFile, 'w') as fp:
        fp.write(command)

    # run mvs-texturing
    print(command)
    os.system(command)

    return

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--projPath', required=True, type=str)
    parser.add_argument('--modelName', required=True, type=str, help='the name of dense model(should be same as sparse model)')
    parser.add_argument('--appPath', required=True, type=str, help='texrecon path')

    parser.add_argument('--imagesDir', default='images')
    parser.add_argument('--denseDir', default='dense')
    parser.add_argument('--mveDataDir', default='mveData')
    parser.add_argument('--scale', default=1.0, type=float)
    parser.add_argument('--interval', default=10, type=int)
    parser.add_argument('--persistOneAtlas', action='store_true')
    parser.add_argument('--plyModelName', default='model_meshed')

    args = parser.parse_args()
    return args

def main():
    args = parseArgs()
    run_mesh_texturing(args)
    return

if __name__ == '__main__':
    main()