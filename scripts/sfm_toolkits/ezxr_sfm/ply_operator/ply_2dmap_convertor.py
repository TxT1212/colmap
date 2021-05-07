import numpy
import cv2
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
from tqdm import tqdm

# resolution = 0.05   2d地图分辨率 单位米
# percent = 1 #95%   边界噪点去除百分比（仅xy轴）
# top_sup_percent = 1  点云密集区域极大值抑制百分比  
# use_z 是否使用z对高度进行筛选（仅保留某一层的地图）

def ply_2dmap(inputfile, outputfile, resolution=0.05, percent=1, top_sup_percent=1, use_z=False, z_min=0.0, z_max=0.0):
  plydata = PlyData.read(inputfile)

  print (plydata['vertex'].dtype)

  x = plydata['vertex']['x'].copy()
  y = plydata['vertex']['y'].copy()
  z = plydata['vertex']['z'].copy()
#   nx = plydata['vertex']['nx'].copy()
#   ny = plydata['vertex']['ny'].copy()
#   nz = plydata['vertex']['nz'].copy()
#   r = plydata['vertex']['red'].copy()
#   g = plydata['vertex']['green'].copy()
#   b = plydata['vertex']['blue'].copy()
#   a = plydata['vertex']['alpha'].copy()

  
#  rmat = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
  xyz = numpy.array([x,y,z])
  xarr = numpy.array([x])
  yarr = numpy.array([y])
  zarr = numpy.array([z])
#   nxarr = numpy.array([nx])
#   nyarr = numpy.array([ny])
#   nzarr = numpy.array([nz])
#   normal = numpy.array([nx,ny,nz])

  xmax = numpy.percentile(xarr, 100-percent)
  xmin = numpy.percentile(xarr, percent)
  ymax = numpy.percentile(yarr, 100-percent)
  ymin = numpy.percentile(yarr, percent)

  x_grid_size = int((xmax - xmin)/resolution) + 1
  y_grid_size = int((ymax - ymin)/resolution) + 1

  map2d=numpy.zeros((x_grid_size,y_grid_size))
#   plt.imshow(map2d)
#   A=numpy.array([[3,2,5],[8,1,2],[6,6,7],[3,5,1]]) #The array to plot
#   im=plt.imshow(A,origin="upper",interpolation="nearest",cmap=plt.cm.gray_r)
#   plt.colorbar(im)

  point_num = xarr.size

  for i in tqdm(range(point_num)):
      if use_z:
          if zarr[0][i] < z_min or zarr[0][i] > z_max:
              continue
      x_grid = int((xarr[0][i] - xmin)/resolution)
      y_grid = int((yarr[0][i] - ymin)/resolution)
      if x_grid < 0 or x_grid >= x_grid_size or y_grid < 0 or y_grid >= y_grid_size:
          continue
      map2d[x_grid][y_grid] = map2d[x_grid][y_grid]+1
  
  mean_gray = map2d.mean()
  print(map2d.mean(), " ", map2d.max())
  
  validnum = numpy.sum(map2d>0)#非0元素的个数 
  validpercent =  100 - 100 * (validnum * top_sup_percent / 100.0) / map2d.size  #压缩前1%的点云密集区域
  
  maxthr = numpy.percentile(map2d,validpercent)
  print("validnum/totalgrid", validnum, "/", map2d.size, " validpercent", validpercent, " maxthr", maxthr)

 # pc = numpy.percentile(map2d, 20)
  map2d[map2d>maxthr] = maxthr

  cv2.imwrite(outputfile, 255.0- (map2d * 255.0 / maxthr))

if __name__ == "__main__":
#hzgb f4 19 7  f2 -4.5, -10

  ply_2dmap("D:\Work\LargeScene\TestScene\guobo\guobo_2k.ply", "F2.png", 0.05, 1, 2, True, -11, -3)
