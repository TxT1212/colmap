import os
import numpy as np
import basicIO as bio
import random

def writeRTFile(filename, R, T, S=None):
    with open(filename, 'w') as fp:
        fp.write('%f %f %f %f %f %f %f %f %f\n' % (
            R[0, 0], R[0, 1], R[0, 2],
            R[1, 0], R[1, 1], R[1, 2],
            R[2, 0], R[2, 1], R[2, 2]
        ))

        fp.write('%f %f %f\n' % (T[0], T[1], T[2]))

        if not (S==None):
            fp.write('%f\n' % S)

def isInlier(srcPoint, tarPoint, R, T, thresh=0.01):
    flag = False
    diffPoint = R.dot(srcPoint) + T - tarPoint
    diffDis = np.linalg.norm(diffPoint)

    if diffDis < thresh:
        flag = True

    return flag, diffDis

def sim3AlignRansac(X, Y, sampleDim=0, goalInlierRatio=0.5, sampleRatio=0.3, maxIterations=100, inlierThresh=0.01, randomSeed=None):
    bestCnt = 0
    bestAlignDis = -1
    bestR = np.identity(3)
    bestT = np.zeros(3, )
    bestS = 1

    if not sampleDim==0:
        X = X.T
        Y = Y.T

    if not (X.shape[0]==3 and Y.shape[0]==3):
        print('X.shape[0] !=3 || Y.shape[0] != 3\n')
        return bestR, bestT, bestS, bestAlignDis

    if not (X.shape[1] == Y.shape[1]):
        print('unmatched sample numbers of X and Y.\n')
        return bestR, bestT, bestS, bestAlignDis

    numTotal = X.shape[1]
    goalInliers = int(numTotal * goalInlierRatio)
    numSamples = int(numTotal * sampleRatio)

    if numSamples < 3:
        print('numSamples < 3\n')
        return bestR, bestT, bestS, bestAlignDis

    bestCnt = 0
    bestAlignDis = -1
    random.seed(randomSeed)
    totalIdx = list(range(numTotal))

    for i in range(maxIterations):
        sampledIdx = random.sample(totalIdx, numSamples)
        R, t, s = sim3Align(X[:, sampledIdx], Y[:, sampledIdx])

        cnt = 0
        alignDisSum = 0
        for j in range(numTotal):
            inlierFlag, alignDis = isInlier(X[:, j], Y[:, j], s*R, t, thresh=inlierThresh)
            if inlierFlag:
                cnt += 1
                alignDisSum += alignDis

        if cnt > bestCnt:
            bestCnt = cnt
            bestR = R
            bestT = t
            bestS = s
            bestAlignDis = alignDisSum/cnt
            if cnt > goalInliers:
                break
    print('took iterations:', i + 1, 'bestAlignDis:', bestAlignDis, 'bestCnt:', bestCnt)
    return bestR, bestT, bestS, bestAlignDis


def sim3Align(X, Y, sampleDim = 0):
    if not sampleDim==0:
        X = X.T
        Y = Y.T

    if not (X.shape[0]==3 and Y.shape[0]==3):
        raise Exception('X.shape[0] !=3 || Y.shape[0] != 3\n')

    if not (X.shape[1] == Y.shape[1]):
        raise Exception('unmatched sample numbers of X and Y.\n')

    # some basic quantity
    numSamples = X.shape[1]
    muX = X.sum(axis=1) / numSamples
    muY = Y.sum(axis=1) / numSamples
    muXTiled = np.tile(muX.reshape(3,1), numSamples)
    sigma2X = np.power(X-muXTiled, 2).sum()/numSamples
    covXY = np.zeros((3, 3))

    for i in range(numSamples):
        cov = np.matmul((Y[:, i] - muY).reshape(3, 1), (X[:, i] - muX).reshape(1, 3))
        covXY += cov

    covXY /= numSamples

    # svd
    U, D, V = np.linalg.svd(covXY)
    r = np.linalg.matrix_rank(covXY)
    S = np.identity(3)

    if np.linalg.det(covXY)<0: S[2,2] = -1

    R = np.matmul(U, np.matmul(S, V))
    diagD = np.diag(D)
    s = np.trace(np.matmul(diagD, S)) / sigma2X
    t = muY - s*R.dot(muX)

    return R, t, s

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

def readRT(filename):
    floatLines = bio.readFloatLines(filename)

    RLine = floatLines[0]

    R = np.array([[RLine[0], RLine[1], RLine[2]],
                  [RLine[3], RLine[4], RLine[5]],
                  [RLine[6], RLine[7], RLine[8]]])

    T = np.array(floatLines[1])

    return R, T

def transformPointsList(pointsList, R, T):
    transformedPoints = []
    for point in pointsList:
        newPoint = R.dot(point) + T
        transformedPoints.append(newPoint)

    return transformedPoints

def readTrajectoryTUM(filename):
    lines = []

    with open(filename) as fp:
        lines = fp.readlines()

    trajLines = []
    for line in lines:
        line = line.strip()

        if len(line) < 1:
            continue

        line_parts = line.split(' ')

        traj = []
        traj.append(line_parts[0])
        for i in range(1, len(line_parts)):
            traj.append(float(line_parts[i]))

        trajLines.append(traj)

    return trajLines

def getMatchedFramePositon(alignTrajList, refTrajList):
    # loca traj name List
    refTrajNameList = []
    for traj in refTrajList:
        refTrajNameList.append(traj[0])

    # match global traj with local traj
    matchedFrameName = []
    matchedAlignFramePos = []
    matchedRefFramePos = []

    for traj in alignTrajList:
        name = traj[0]

        refIdx = -1
        if name in refTrajNameList:
            refIdx = refTrajNameList.index(name)

        if refIdx >= 0:
            matchedFrameName.append(name)

            # align pos WC
            alignTWC = [traj[1], traj[2], traj[3]]
            matchedAlignFramePos.append(alignTWC)

            # ref pose WC
            refTraj = refTrajList[refIdx]
            refTWC = [refTraj[1], refTraj[2], refTraj[3]]
            matchedRefFramePos.append(refTWC)

    return matchedFrameName, matchedAlignFramePos, matchedRefFramePos

def accuAdjTrans2Global(RList, TList):
    numParts = len(RList)

    globalRList = []
    globalTList = []
    for i in range(numParts):
        if i == 0:
            globalRList.append(RList[i])
            globalTList.append(TList[i])
        else:
            globalT = (globalRList[i-1].dot(TList[i]) + globalTList[i-1]).copy()
            globalR = (globalRList[i-1].dot(RList[i])).copy()

            globalRList.append(globalR)
            globalTList.append(globalT)

    return globalRList, globalTList

if __name__ == "__main__":
    workDir = "E:/lzx-data/TUM/rgbd_dataset_freiburg1_room"
    trajFileName = "CameraTrajectory"

    trajFile = os.path.join(workDir, trajFileName+".txt")
    trajRotMatFile = os.path.join(workDir, trajFileName + "_rotmat.txt")

