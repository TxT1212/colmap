# coding: utf-8
import os
import sys
import struct
import argparse
import numpy as np

sys.path.append("../")

from colmap_process.colmap_read_write_model import *
from colmap_process.colmap_export_geo import *

def read_image_model(path, ext):
    if ext == ".txt":
        images = read_images_text(os.path.join(path, "images" + ext))
    else:
        images = read_images_binary(os.path.join(path, "images" + ext))
    return images 

def model2tum(images, tum_path):
   # sorted = {}
    sortimg = {k: v for k, v in sorted(images.items(), key=lambda item: item[1])}
    # for _, img in images.items():
    #     img_idx = (int)(os.path.splitext(os.path.basename(img.name))[0])
    #     img.id = img_idx
    #     sorted[img_idx] = img
    
    with open(tum_path, "w") as fid:
        for _, img in sortimg.items():
            print (img.name)
            rmat = quaternion_matrix(img.qvec).T
            tvec = img.tvec
            tnew = -rmat @ tvec
            qnew = quaternion_from_matrix(rmat)
            fid.write(img.name.split('.')[0] + " " + str(tnew[0]) + " " + str(tnew[1]) + " " + str(tnew[2]) \
                + " " + str(qnew[1]) + " " + str(qnew[2]) + " " + str(qnew[3]) + " " + str(qnew[0]) + "\n" )

 #   return images

def main():
    parser = argparse.ArgumentParser(description='Convert orbslam traj to colmap image model')
    parser.add_argument('--input', help='colmap model path')
    parser.add_argument('--ext', default=".txt")
    parser.add_argument('--output', help='tum_traj')
    args = parser.parse_args()
    images = read_image_model(args.input, args.ext)

    model2tum(images, args.output)
    #write_model(cameras, images, points3D, path=args.output_model, ext=args.output_format, poseonly=args.poseonly)

if __name__ == "__main__":
    main()