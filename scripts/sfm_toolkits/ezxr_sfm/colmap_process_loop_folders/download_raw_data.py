# coding: utf-8
import os
import argparse
import json
import urllib.request
import time

def parseRecordJson(jsonFile):
    recordDict = {}

    if not os.path.isfile(jsonFile):
        print('record file does not exist: %s\n' % jsonFile)
        return recordDict

    with open(jsonFile, 'r', encoding='UTF-8') as fp:
        recordDict = json.load(fp)

    return recordDict

def parseRouteInfoJson(jsonFile):
    routeInfoList = []

    if not os.path.isfile(jsonFile):
        print('route info file does not exist: %s\n' % jsonFile)
        return routeInfoList

    with open(jsonFile, 'r', encoding='UTF-8') as fp:
        jsonValue = json.load(fp)

    if jsonValue['success']:
        routeInfoList = jsonValue['data']['routes']

    return routeInfoList

def deleteSameNameFiles(videosPath, tarFileName):
    tarName, _ = os.path.splitext(tarFileName)
    srcFiles = os.listdir(videosPath)

    for fileName in srcFiles:
        fullFile = os.path.join(videosPath, fileName)
        if os.path.isfile(fullFile):
            name, _ = os.path.splitext(fileName)
            if name == tarName:
                os.remove(fullFile)
    return

def getCorrKeyUnique(tarValue, curDict):

    tmpKeys = []
    for key, value in curDict.items():
        if value==tarValue:
            tmpKeys.append(key)

    corrKey=None
    if len(tmpKeys)>1:
        raise Exception('multiple keys with same value')
    elif len(tmpKeys)==1:
        corrKey = tmpKeys[0]
    else:
        corrKey=None

    return corrKey

def checkAndBackupFile(tarFilePath, downloadedDict):
    filePath, filename = os.path.split(tarFilePath)
    name, ext = os.path.splitext(filename)

    corrKey = getCorrKeyUnique(filename, downloadedDict)

    if corrKey==None:
        if os.path.isfile(tarFilePath):
            print('%s is not recorded in downloadedDict and its filename is supposed to be occupied by another file. So we have to deltete it.' % filename)
            os.remove(tarFilePath)
    else:
        bkpStr = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
        newTarFilename = name + '_' + bkpStr + ext
        newTarFilepath = os.path.join(filePath, newTarFilename)

        print('%s is recorded in downloadedDict and its filename is supposed to be occupied by another file.\n So we have to rename it as %s' % (filename, newTarFilename))
        
        if os.path.isfile(tarFilePath):
            os.rename(tarFilePath, newTarFilepath)
            downloadedDict[corrKey] = newTarFilename
        else:
            raise Exception('However, we could not find it in videos dir, thus we could not conduct the remane operation either.')

    return downloadedDict

def downloadRouteData(routeInfo, videosPath, configPath,
                      downloadedDict={},
                      validFileExtensions=[],
                      failedUrlList=[],
                      triedUrlList=[],
                      maxTryTime=2,
                      tarPlaces=[]):
    projName = routeInfo['projName']
    sceneName = routeInfo['sceneName']
    placeName = routeInfo['placeName']
    routeName = routeInfo['routeName']

    if (len(tarPlaces) > 0) and (not (placeName in tarPlaces)):
        return downloadedDict, failedUrlList, triedUrlList

    routeValidSeqsDict = {}
    routeValidSeqsDict['validSeqs'] = []
    for seqInfo in routeInfo['validSeqs']:
        batchNameOnSeq = seqInfo['bn']
        fileUrl = seqInfo['fileUrl']

        _, fileName = os.path.split(fileUrl)
        _, ext = os.path.splitext(fileName)

        tarFileName = projName + '_' + sceneName + '_' + batchNameOnSeq + '_' + placeName + '_' + routeName + '_' + \
                      fileName[0:6] + ext
        tarFilePath = os.path.join(videosPath, tarFileName)

        if len(validFileExtensions)>0 and (not ext in validFileExtensions):
            print('warning!!! unsupported file extension detected: %s\n' % ext)
            continue

        if len(downloadedDict.keys()) > 0 and (fileName in downloadedDict.keys()):
            recordFilePath = os.path.join(videosPath, downloadedDict[fileName])
            if os.path.isfile(recordFilePath):
                print('%s (in url) has already been in downloadedDict, corresponding with %s (in videos dir)\n' %
                      (fileName, downloadedDict[fileName]))

                if not(downloadedDict[fileName] == tarFileName):
                    downloadedDict = checkAndBackupFile(tarFilePath, downloadedDict)
                    print('rename %s as %s\n' % (downloadedDict[fileName], tarFileName))
                    os.rename(recordFilePath, tarFilePath)
            else:
                print('warning!!! According to config/downloadRecord.json, %s (in url) has already been downloaded, corresponding with %s (in videos dir).'
                      'However, we actually could not find it.\n' %
                      (fileName, downloadedDict[fileName]))
                if not (downloadedDict[fileName] == tarFileName):
                    raise Exception('More Seriously, we found the already downloaded file %s should be renamed as %s, '
                                    'but we could not make it unless it exists in videos dir.' % (downloadedDict[fileName], tarFileName))

            downloadedDict[fileName] = tarFileName
            routeValidSeqsDict['validSeqs'].append(tarFileName)
        else:
            downloadedDict = checkAndBackupFile(tarFilePath, downloadedDict)
            triedUrlList.append(fileUrl)

            print('download file from %s to %s...\n' % (fileUrl, tarFilePath))
            tryCnt = 0
            while True:
                try:
                    print('try_count = %d\n' % (tryCnt + 1))
                    urllib.request.urlretrieve(fileUrl, filename=tarFilePath)
                    downloadedDict[fileName] = tarFileName
                    routeValidSeqsDict['validSeqs'].append(tarFileName)
                    print('succeed.\n')
                    break
                except Exception as e:
                    tryCnt += 1
                    print('failed with message:')
                    print(e)
                    if tryCnt==maxTryTime:
                        failedUrlList.append(fileUrl)
                        print("After %d times tried, error still occurred when downloading file" % maxTryTime)
                        break

    routeValidSeqsJsonFile = os.path.join(configPath,
                                          sceneName + '_' + placeName + '_' + routeName + '.json')

    with open(routeValidSeqsJsonFile, 'w') as fp:
        json.dump(routeValidSeqsDict, fp, indent=4)

    return downloadedDict, failedUrlList, triedUrlList, routeValidSeqsDict['validSeqs']

def readValidStrLines(filePath):
    lines = []
    with open(filePath) as fp:
        lines = fp.readlines()

    validLines = []
    for line in lines:
        line = line.strip()
        if len(line)>0:
            validLines.append(line)

    return validLines

def run_raw_data_downloader(args):
    downloadedListFileName = 'downloadRecord.json'

    videosPath = os.path.join(args.projPath, "videos")
    if not os.path.isdir(videosPath):
        os.mkdir(videosPath)

    configPath = os.path.join(args.projPath, "config")
    if not os.path.isdir(configPath):
        os.mkdir(configPath)

    # get list of already downloaded videos
    downloadedListFile = os.path.join(configPath, downloadedListFileName)
    downloadedDict = parseRecordJson(downloadedListFile)

    # get list of route info dict
    routeInfoList = parseRouteInfoJson(args.routeJson)

    # process each route
    if not (args.extensionFile==None):
        validFileExtensions = readValidStrLines(args.extensionFile)
    else:
        validFileExtensions = ['.mp4', '.avi', '.MOV', '.MP4', '.AVI', '.mov', '.insv', '.zip']

    failedUrlList = []
    triedUrlList = []
    allValidSeqs = []
    for routeInfo in routeInfoList:
        downloadedDict, failedUrlList, triedUrlList, routeValidSeqs = downloadRouteData(routeInfo, videosPath, configPath,
                          downloadedDict=downloadedDict,
                          validFileExtensions=validFileExtensions,
                          failedUrlList=failedUrlList,
                          triedUrlList=triedUrlList,
                          maxTryTime=args.maxTryTime,
                          tarPlaces=args.tarPlaces)
        
        allValidSeqs += routeValidSeqs

    # update downloadedListFile
    with open(downloadedListFile, 'w') as fp:
        json.dump(downloadedDict, fp, indent=4)
    
    allSeqsListFile = os.path.join(configPath, 'allSeqs.txt')
    with open(allSeqsListFile, 'w') as fp:
        for seq in allValidSeqs:
            seqName, _ = os.path.splitext(seq)
            fp.write(seqName + '\n')

    if len(failedUrlList) > 0:
        print('download incomplete, %d urls failed, please try again: \n'%len(failedUrlList))
        for url in failedUrlList:
            print(url + '\n')
    else:
        print('download complete.\n')

    return

def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--projPath', required=True, type=str)
    parser.add_argument('--routeJson', required=True, type=str, help='full path of routejson file')
    parser.add_argument('--extensionFile', default=None, type=str, help="full path of list file of supported video extensions")
    parser.add_argument('--maxTryTime', default=2, help='the max number of download try for each url')
    parser.add_argument('--tarPlaces', type=str, nargs = '+', default=[])

    args = parser.parse_args()
    return args

def main():
    args = parseArgs()
    run_raw_data_downloader(args)

if __name__ == '__main__':
    main()