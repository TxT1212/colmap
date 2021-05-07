# coding: utf-8
import sys
import sqlite3
import math
import argparse
import numpy as np

sys.path.append('../')
from shutil import copyfile
from colmap_process.colmap_read_write_model import *
from colmap_process.colmap_model_modify import *

def rescale_image_model_binary(old_model_path, scale):
  images_model = read_images_binary(old_model_path + "/images.bin")
  images_rescale = {}

  for _, image in images_model.items():
    image_id = image.id
    qvec = image.qvec
    tvec = image.tvec * scale
    camera_id = image.camera_id
    image_name = image.name
    xys = image.xys
    point3D_ids=image.point3D_ids
    images_rescale[image_id] = Image(
          id=image_id, qvec=qvec, tvec=tvec,
          camera_id=camera_id, name=image_name,
          xys=xys, point3D_ids=point3D_ids)
    
  return images_rescale

def rescale_points_model_binary(old_model_path, scale):
  points_model = read_points3d_binary(old_model_path + "/points3D.bin")
  points_rescale = {}

  for _, points in points_model.items():
    point3D_id = points.id
    xyz = points.xyz * scale
    rgb = points.rgb
    error = points.error
    image_ids=points.image_ids
    point2D_idxs=points.point2D_idxs

    points_rescale[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs)
  return points_rescale

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', required=True)
    parser.add_argument('--input_model', required=True)
    parser.add_argument('--output_model', required=True)
    
    args = parser.parse_args()
    return args

def rescale_model(scale, input_model, output_model):
    images_rescale = rescale_image_model_binary(input_model, scale)
    points_rescale = rescale_points_model_binary(input_model, scale)
    if not os.path.exists(output_model):
      os.makedirs(output_model)
    write_images_binary(images_rescale, output_model + "/images.bin")
    write_points3d_binary(points_rescale, output_model + "/points3D.bin")
    if not (input_model == output_model):
        copyfile( input_model + "/cameras.bin", output_model + "/cameras.bin")
    write_geo_text(images_rescale, output_model + "/geos.txt")

def main():
    args = parse_args()
    scale = float(args.scale)
    images_rescale = rescale_image_model_binary(args.input_model, scale)
    points_rescale = rescale_points_model_binary(args.input_model, scale)

    if not os.path.exists(args.output_model):
      os.makedirs(args.output_model)
    write_images_binary(images_rescale, args.output_model + "/images.bin")
    write_points3d_binary(points_rescale, args.output_model + "/points3D.bin")
    copyfile( args.input_model + "/cameras.bin", args.output_model + "/cameras.bin")
    write_geo_text(images_rescale, args.output_model + "/geos.txt")

if __name__ == '__main__':
    main()