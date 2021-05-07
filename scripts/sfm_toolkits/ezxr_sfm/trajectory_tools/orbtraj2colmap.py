# coding: utf-8
import os
import sys
import struct
import argparse
import numpy as np

sys.path.append("../")

from colmap_process.colmap_read_write_model import *
from colmap_process.colmap_export_geo import *

def read_orb_traj_text(traj_path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    image_id = 0
    with open(traj_path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_name = elems[0] + ".png"
                tvec = np.array(tuple(map(float, elems[1:4])))
                qvec = np.array(tuple(map(float, elems[4:8])))
                qvec_new = [qvec[3], qvec[0], qvec[1], qvec[2]]
                qvec = qvec_new

                rmat = quaternion_matrix(qvec).T
                tnew = -rmat @ tvec

                qnew = quaternion_from_matrix(rmat)

                qvec = qnew
                tvec = tnew

                camera_id = 2048
               # image_name = elems[9]
                # elems = fid.readline().split()
                # xys = np.column_stack([tuple(map(float, elems[0::3])),
                #                        tuple(map(float, elems[1::3]))])
                # point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=[], point3D_ids=[])
                image_id = image_id +1
    return images

def main():
    parser = argparse.ArgumentParser(description='Convert orbslam traj to colmap image model')
    parser.add_argument('--traj', help='txt trajectory file')
    parser.add_argument('--output', help='colmap image model')
    args = parser.parse_args()
    images = read_orb_traj_text(args.traj)

    write_images_text(images, args.output, False)
    #write_model(cameras, images, points3D, path=args.output_model, ext=args.output_format, poseonly=args.poseonly)

if __name__ == "__main__":
    main()
