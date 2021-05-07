# coding: utf-8
import numpy as np
import argparse
from plyfile import PlyData, PlyElement

def transform_ply(inputfile, outputfile, transform_mat):
  plydata = PlyData.read(inputfile)

  x = plydata['vertex']['x'].copy()
  y = plydata['vertex']['y'].copy()
  z = plydata['vertex']['z'].copy()
  nx = plydata['vertex']['nx'].copy()
  ny = plydata['vertex']['ny'].copy()
  nz = plydata['vertex']['nz'].copy()
  
  rmat = transform_mat[0:3, 0:3]
  tvec = transform_mat[0:3, 3]

  xyz = np.array([x,y,z])
  res_xyz = np.matmul(rmat, xyz) + tvec.reshape(3,1)

  normal = np.array([nx,ny,nz])
  res_normal = np.matmul(rmat, normal)

  plydata['vertex']['x'] = res_xyz[0,:]
  plydata['vertex']['y'] = res_xyz[1,:]
  plydata['vertex']['z'] = res_xyz[2,:]

  plydata['vertex']['nx'] = res_normal[0,:]
  plydata['vertex']['ny'] = res_normal[1,:]
  plydata['vertex']['nz'] = res_normal[2,:]

  PlyData.write(plydata, outputfile)

def read_transform_mat(transform_txt_path_name):
  mat_str = []
  for line in open(transform_txt_path_name):
    li = line.strip()
    if not li.startswith("#"):
      mat_str.append(li)
  print(mat_str)
  aa = list(map(lambda mat_str: float(mat_str), list_of_string_floats))
  print(aa)

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--src_ply', required=True)
  parser.add_argument('--transform_txt', required=True)
  args = parser.parse_args()
  return args

if __name__ == "__main__":
  args = parse_args()
  outply_path_name = args.src_ply[0:-4] + '_transformed.ply'
  transform_mat = np.loadtxt(args.transform_txt, comments='#')
  transform_ply(args.src_ply, outply_path_name, transform_mat)