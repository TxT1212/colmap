
from __future__ import print_function
 
import numpy as np
from cv2 import cv2 as cv 

import argparse
import sys
import os

modes = (cv.Stitcher_PANORAMA, cv.Stitcher_SCANS)

parser = argparse.ArgumentParser(prog='stitching.py', description='Stitching sample.')
parser.add_argument('--mode',
    type = int, choices = modes, default = cv.Stitcher_PANORAMA,
    help = 'Determines configuration of stitcher. The default is `PANORAMA` (%d), '
        'mode suitable for creating photo panoramas. Option `SCANS` (%d) is suitable '
        'for stitching materials under affine transformation, such as scans.' % modes)
parser.add_argument('--output', default = 'result.jpg',
    help = 'Resulting image. The default is `result.jpg`.')
parser.add_argument('--img', nargs='+', help = 'input images')

#__doc__ += '\n' + parser.format_help()
def getFiles(dir, suffix):
    res = []
    for root, directory, files in os.walk(dir):
        for filename in files:
            name, suf = os.path.splitext(filename)
            if suf == suffix:
                res.append(os.path.join(root, filename))
    return res

def main():
    args = parser.parse_args()

    # read input images
    imgs = []
    for img_path in args.img:
        files = getFiles(img_path, ".png")
        for img_name in files:
            print(img_name)
            img = cv.imread(cv.samples.findFile(img_name))
            if img is None:
                print("can't read image " + img_name)
                sys.exit(-1)
            imgs.append(img)

    stitcher = cv.Stitcher.create(args.mode)
    status, pano = stitcher.stitch(imgs)

    if status != cv.Stitcher_OK:
        print("Can't stitch images, error code = %d" % status)
        sys.exit(-1)

    cv.imwrite(args.output, pano)
    print("stitching completed successfully. %s saved!" % args.output)

    print('Done')


if __name__ == '__main__':
    print("QAQ")
    main()
    cv.destroyAllWindows()