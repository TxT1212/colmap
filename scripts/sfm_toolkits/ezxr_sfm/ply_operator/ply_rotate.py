# coding: utf-8
import numpy
from plyfile import PlyData, PlyElement

def rotate_ply(inputfile, outputfile, rmat):
  plydata = PlyData.read(inputfile)

  x = plydata['vertex']['x'].copy()
  y = plydata['vertex']['y'].copy()
  z = plydata['vertex']['z'].copy()
  nx = plydata['vertex']['nx'].copy()
  ny = plydata['vertex']['ny'].copy()
  nz = plydata['vertex']['nz'].copy()
  
#  rmat = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
  xyz = numpy.array([x,y,z])
  normal = numpy.array([nx,ny,nz])
  
  res_xyz = rmat @ xyz
  res_normal = rmat @ normal

  plydata['vertex']['x'] = res_xyz[0,:]
  plydata['vertex']['y'] = res_xyz[1,:]
  plydata['vertex']['z'] = res_xyz[2,:]

  plydata['vertex']['nx'] = res_normal[0,:]
  plydata['vertex']['ny'] = res_normal[1,:]
  plydata['vertex']['nz'] = res_normal[2,:]

  print("Write to ", outputfile, "\nwith rotation matrix")
  print(rmat)
  PlyData.write(plydata, outputfile)


if __name__ == "__main__":
  # #西湖
  #   mmm = [     \
  #       [0.999104,-0.0248649,-0.0342417],\
  #       [0.0226688,0.99775,-0.0630934],\
  #       [0.0357334,0.0622606,0.99742]\
  #   ]
  #金鱼
  #   0.999848   0.0152271  -0.0084609
# -0.00885331   0.0258903   -0.999626
#  -0.0150023    0.999549   0.0260212

    mmm = [     \
        [0.999848,0.0152271,-0.0084609],\
        [-0.0150023,0.999549,0.0260212],\
        [0.00885331,-0.0258903,0.999626]
    ]

    rotate_ply("/media/netease/Storage/LargeScene/Scene/XixiWetland/models/goldfish_200828/goldfish_200828_cut_simp.ply", 
    "/media/netease/Storage/LargeScene/Scene/XixiWetland/models/goldfish_200828/goldfish_200828_cut_simp_rot.ply", mmm)