# coding: utf-8
import numpy
from plyfile import PlyData, PlyElement

def ply_autocut(inputfile, outputfile, percent):
  plydata = PlyData.read(inputfile)

  print (plydata['vertex'].dtype)

  x = plydata['vertex']['x'].copy()
  y = plydata['vertex']['y'].copy()
  z = plydata['vertex']['z'].copy()
  nx = plydata['vertex']['nx'].copy()
  ny = plydata['vertex']['ny'].copy()
  nz = plydata['vertex']['nz'].copy()
  r = plydata['vertex']['red'].copy()
  g = plydata['vertex']['green'].copy()
  b = plydata['vertex']['blue'].copy()
  a = plydata['vertex']['alpha'].copy()

  #  rmat = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
  xyz = numpy.array([x,y,z])
  xarr = numpy.array([x])
  yarr = numpy.array([y])
  zarr = numpy.array([z])
  nxarr = numpy.array([nx])
  nyarr = numpy.array([ny])
  nzarr = numpy.array([nz])
  normal = numpy.array([nx,ny,nz])

  xmax = numpy.percentile(xarr, 100-percent)
  xmin = numpy.percentile(xarr, percent)
  ymax = numpy.percentile(yarr, 100-percent)
  ymin = numpy.percentile(yarr, percent)
  zmax = numpy.percentile(zarr, 100-percent)
  zmin = numpy.percentile(zarr, percent)

  idx = numpy.where(\
    (xarr >= xmin) & (xarr <= xmax) & \
    (yarr >= ymin) & (yarr <= ymax) & \
    (zarr >= zmin) & (zarr <= zmax))
  idx = idx[1]
 # print(idx)
  
  # res_xyz = numpy.array([xarr[idx],yarr[idx],zarr[idx]])
  # res_normal = numpy.array([nxarr[idx],nyarr[idx],nzarr[idx]])

  vertex_all = numpy.zeros(idx.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), ('red', 'uint8'), ('green', 'uint8'), ('blue', 'uint8'), ('alpha', 'uint8')])

#  vertex = numpy.array(res_xyz.shape[0], dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
  # vertex = numpy.zeros(res_xyz.shape[1], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
  # normply = numpy.zeros(res_xyz.shape[1], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
  for i in range(idx.shape[0]):
    vertex_all[i] = plydata['vertex'][idx[i]]
  #  vertex_all[i] = (res_xyz[0][i], res_xyz[1][i], res_xyz[2][i], res_normal[0][i], res_normal[1][i], res_normal[2][i], r[i], g[i], b[i], a[i])
  #  normply[i] = (res_normal[0][i], res_normal[1][i], res_normal[2][i])
  # el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
  points = PlyElement.describe(vertex_all, 'vertex', comments=['vertices']) #PlyElement.describe(np.array(res_xyz,dtype=[......]),'vertex')
 # normals = PlyElement.describe(normply,'normals')
  # plydata['vertex']['x'] = res_xyz[0,:]
  # plydata['vertex']['y'] = res_xyz[1,:]
  # plydata['vertex']['z'] = res_xyz[2,:]

  # plydata['vertex']['nx'] = res_normal[0,:]
  # plydata['vertex']['ny'] = res_normal[1,:]
  # plydata['vertex']['nz'] = res_normal[2,:]

  print("Write to ", outputfile, "\nwith rotation matrix")

 # points.write(outputfile)

  PlyData([points]).write(outputfile)
 # PlyData.write(plydata, outputfile)


def cut_ply(inputfile, outputfile, axis, lower, upper):
  plydata = PlyData.read(inputfile)

  print (plydata['vertex'].dtype)

  x = plydata['vertex']['x'].copy()
  y = plydata['vertex']['y'].copy()
  z = plydata['vertex']['z'].copy()
  nx = plydata['vertex']['nx'].copy()
  ny = plydata['vertex']['ny'].copy()
  nz = plydata['vertex']['nz'].copy()
  r = plydata['vertex']['red'].copy()
  g = plydata['vertex']['green'].copy()
  b = plydata['vertex']['blue'].copy()
  a = plydata['vertex']['alpha'].copy()

  
#  rmat = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
  xyz = numpy.array([x,y,z])
  xarr = numpy.array([x])
  yarr = numpy.array([y])
  zarr = numpy.array([z])
  nxarr = numpy.array([nx])
  nyarr = numpy.array([ny])
  nzarr = numpy.array([nz])
  normal = numpy.array([nx,ny,nz])

  idx = numpy.where((zarr >= lower) & (zarr <= upper))
  idx = idx[1]
 # print(idx)
  
  # res_xyz = numpy.array([xarr[idx],yarr[idx],zarr[idx]])
  # res_normal = numpy.array([nxarr[idx],nyarr[idx],nzarr[idx]])

  vertex_all = numpy.zeros(idx.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), ('red', 'uint8'), ('green', 'uint8'), ('blue', 'uint8'), ('alpha', 'uint8')])

#  vertex = numpy.array(res_xyz.shape[0], dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
  # vertex = numpy.zeros(res_xyz.shape[1], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
  # normply = numpy.zeros(res_xyz.shape[1], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
  for i in range(idx.shape[0]):
    vertex_all[i] = plydata['vertex'][idx[i]]
  #  vertex_all[i] = (res_xyz[0][i], res_xyz[1][i], res_xyz[2][i], res_normal[0][i], res_normal[1][i], res_normal[2][i], r[i], g[i], b[i], a[i])
  #  normply[i] = (res_normal[0][i], res_normal[1][i], res_normal[2][i])
  # el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
  points = PlyElement.describe(vertex_all, 'vertex', comments=['vertices']) #PlyElement.describe(np.array(res_xyz,dtype=[......]),'vertex')
 # normals = PlyElement.describe(normply,'normals')
  # plydata['vertex']['x'] = res_xyz[0,:]
  # plydata['vertex']['y'] = res_xyz[1,:]
  # plydata['vertex']['z'] = res_xyz[2,:]

  # plydata['vertex']['nx'] = res_normal[0,:]
  # plydata['vertex']['ny'] = res_normal[1,:]
  # plydata['vertex']['nz'] = res_normal[2,:]

  print("Write to ", outputfile, "\nwith rotation matrix")

 # points.write(outputfile)

  PlyData([points]).write(outputfile)
 # PlyData.write(plydata, outputfile)


if __name__ == "__main__":

    cut_ply("/media/netease/Dataset/LargeScene/Scene/Wanxiangcheng/pointclouds/F2_dense_gz_f2_cut.ply", 
    "/media/netease/Dataset/LargeScene/Scene/Wanxiangcheng/pointclouds/F2_dense_gz_f2_cut_2.ply", "z", 3, 8.0)