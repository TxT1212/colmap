# coding: utf-8
## 修改txt、bin model中的image对应的iamge_id，以和db对应
## point中的序号暂时不动，相机序号保持不变

## image_id db的 Camera_id db的 point 删了 
## 也就是只有位姿是原来model的

import sys
import sqlite3
import math
import copy
import os
import numpy as np
sys.path.append('../')
from colmap_process.colmap_read_write_model import *
from colmap_process.colmap_db_parser import *
from colmap_process.colmap_keyframe_selecter import auto_read_model
#from scipy.spatial.transform import Rotation as R

def read_cam_model_from_db(database_path):
  db = COLMAPDatabase.connect(database_path)
  db.create_tables()
  cameras_db = db.execute("SELECT * FROM cameras")

  cameras = {}
  for row in cameras_db:
    camera_id = row[0]
    model = row[1]
    model_name = CAMERA_MODEL_IDS[model].model_name
    width = row[2]
    height = row[3]
    params = row[4]
    params = blob_to_array(params, np.float64)
    cameras[camera_id] = Camera(id=camera_id,
                                model=model_name,
                                width=width,
                                height=height,
                                params=params)

  return cameras
 # write_cameras_text(cameras, new_cam_model)

def quaternion_matrix(quaternion):
  # """Return homogeneous rotation matrix from quaternion.

  # >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
  # >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
  # True
  # >>> M = quaternion_matrix([1, 0, 0, 0])
  # >>> numpy.allclose(M, numpy.identity(4))
  # True
  # >>> M = quaternion_matrix([0, 1, 0, 0])
  # >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
  # True

  # """
  q = np.array(quaternion, dtype=np.float64, copy=True)
  n = np.dot(q, q)
  if n < 1e-9:
      return np.identity(3)
  q *= math.sqrt(2.0 / n)
  q = np.outer(q, q)
  return np.array([
      [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0]],
      [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0]],
      [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]]
      ])

def write_geo_text(images, path):
  if len(images) == 0:
      mean_observations = 0
  else:
      mean_observations = sum((len(img.point3D_ids) for _, img in images.items()))/len(images)
  HEADER = '# Image list with two lines of data per image:\n'
  '#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n'
  '#   POINTS2D[] as (X, Y, POINT3D_ID)\n'
  '# Number of images: {}, mean observations per image: {}\n'.format(len(images), mean_observations)

  with open(path, "w") as fid:
     # fid.write(HEADER)
    for _, img in images.items():
  #    rmat = R.from_quat(img.qvec)
      rmat = quaternion_matrix(img.qvec).T
      tvec = img.tvec
      tnew = -rmat @ tvec
      image_header = [img.name, *tnew]
      first_line = " ".join(map(str, image_header))
      fid.write(first_line + "\n")

def update_loc_model_id_refer_to_map_model_id(loc_model_path, map_model_path, new_loc_model_path):
  '''
  背景: 使用model_merger的时候，colmap是假设两个model都来自同一个database，因此common-image的id是一样的
  但是当我们把loc和map合并成locmap的时候，loc里的image-id跟map里的同一张image-id就不对应了，会报错
  
  函数说明: 根据map-model修改loc-model里的image-id和points3d对应的image-id
  '''
  if not os.path.isdir(loc_model_path):
    print('ERROR! invalid path:', loc_model_path)
    exit(0)
  if not os.path.isdir(map_model_path):
    print('ERROR! invalid path:', map_model_path)
    exit(0)
  if loc_model_path[-1] != '/':
    loc_model_path = loc_model_path + '/'
  if new_loc_model_path[-1] != '/':
    new_loc_model_path = new_loc_model_path + '/'
  assert(loc_model_path != new_loc_model_path)
  if not os.path.isdir(new_loc_model_path):
    os.mkdir(new_loc_model_path)
  # 读loc-model，需要修改它，使得它跟map-model的id对应，正常情况下locmap-db和locmap-model的id也是对应的
  loc_cameras, loc_images, loc_points3d = auto_read_model(loc_model_path)
  map_cameras, map_images, map_points3d = auto_read_model(map_model_path)
  
  # map先做dict，key=image_name, value=[image_id, camera_id]
  map_name_id_dict = {} 
  for _, image in map_images.items():
    map_name_id_dict[image.name] = [image.id, image.camera_id]
  
  # 生成新的model
  new_loc_cameras = {}
  new_loc_images = {}
  new_loc_points3d = {}

  '''
  Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
  BaseImage = collections.namedtuple(
      "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
  Point3D = collections.namedtuple(
      "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
  '''
  # 循环loc-model的images
  print('modify cameras and images...')
  for _, image in loc_images.items():
    assert(image.name in map_name_id_dict.keys()) # map肯定要包含loc的所有images
    image_id_in_map = map_name_id_dict[image.name][0]
    camera_id_in_map = map_name_id_dict[image.name][1]

    # 修改camera，只修改id，其他不变
    # 这里循环的是images，所以camera的id是会重复的
    # 不同的image共享一个camera，所以先判断
    if not camera_id_in_map in new_loc_cameras.keys():
      old_camera = loc_cameras[image.camera_id]
      new_camera = Camera(camera_id_in_map, old_camera.model, old_camera.width, old_camera.height, old_camera.params)
      new_loc_cameras[new_camera.id] = new_camera

    # 修改image，只修改id，其他不变
    # image的id是不会重复的，所以直接添加
    new_image = Image(image_id_in_map, image.qvec, image.tvec, camera_id_in_map, image.name, image.xys, image.point3D_ids)
    new_loc_images[new_image.id] = new_image
  # assert(len(loc_cameras) == len(new_loc_cameras))
  assert(len(loc_images) == len(new_loc_images))
  print('modify points3d...')
  # 循环loc-model的points3d
  for _, point3d in loc_points3d.items():
    # 修改points3d，只修改它对应的image_ids，其他不变
    new_image_ids = point3d.image_ids
    for idx in range(len(point3d.image_ids)):
      image_id = point3d.image_ids[idx]
      image_name = loc_images[image_id].name
      image_id_in_map = map_name_id_dict[image_name][0]
      new_image_ids[idx] = image_id_in_map
    new_point3d = Point3D(point3d.id, point3d.xyz, point3d.rgb, point3d.error, new_image_ids, point3d.point2D_idxs)
    new_loc_points3d[new_point3d.id] = new_point3d
  assert(len(loc_points3d) == len(new_loc_points3d))
  print('write new model...')
  # 写新的model
  write_model(new_loc_cameras, new_loc_images, new_loc_points3d, new_loc_model_path, '.bin')
  return    

def update_loc_model_id_refer_to_locmap_database(loc_model_path, locmap_database_path, new_loc_model_path):
  '''
  背景: 使用model_merger的时候，colmap是假设两个model都来自同一个database，因此common-image的id是一样的
  但是当我们把loc和map合并成locmap的时候，loc里的image-id跟locmap里的同一张image-id就不对应了，会报错
  
  函数说明: 根据locmap的database修改loc-model里的image-id和points3d对应的image-id
  '''
  if not os.path.isdir(loc_model_path):
    print('ERROR! invalid path:', loc_model_path)
    exit(0)
  if not os.path.isfile(locmap_database_path):
    print('ERROR! invalid path:', locmap_database_path)
    exit(0)
  if loc_model_path[-1] != '/':
    loc_model_path = loc_model_path + '/'
  if new_loc_model_path[-1] != '/':
    new_loc_model_path = new_loc_model_path + '/'
  assert(loc_model_path != new_loc_model_path)
  if not os.path.isdir(new_loc_model_path):
    os.mkdir(new_loc_model_path)
  # 读loc-model，需要修改它，使得它跟locmap-db的id对应，正常情况下locmap-db和locmap-model的id也是对应的
  loc_cameras, loc_images, loc_points3d = auto_read_model(loc_model_path)
  
  # 访问db，把正确的id读出来
  db = COLMAPDatabase.connect(locmap_database_path)
  cursor = db.cursor()
  cursor.execute("select * from images")
  imgs = cursor.fetchall()
  db.close() # 读完db早点关了它
  
  # locmap先做dict，key=image_name, value=[image_id, camera_id]
  locmap_name_id_dict = {} 
  for img in imgs:
    image_id = img[0]
    image_name = img[1]
    camera_id = img[2]
    locmap_name_id_dict[image_name] = [image_id, camera_id]
  
  # 生成新的model
  new_loc_cameras = {}
  new_loc_images = {}
  new_loc_points3d = {}

  '''
  Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
  BaseImage = collections.namedtuple(
      "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
  Point3D = collections.namedtuple(
      "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
  '''
  # 循环loc-model的images
  print('modify cameras and images...')
  for _, image in loc_images.items():
    assert(image.name in locmap_name_id_dict.keys()) # locmap的db里肯定有loc的所有图像，否则db-merge就出错了
    image_id_in_locmap = locmap_name_id_dict[image.name][0]
    camera_id_in_locmap = locmap_name_id_dict[image.name][1]

    # 修改camera，只修改id，其他不变
    # camera的id是会重复的，不同的image共享一个camera，所以先判断
    if not camera_id_in_locmap in new_loc_cameras.keys():
      old_camera = loc_cameras[image.camera_id]
      new_camera = Camera(camera_id_in_locmap, old_camera.model, old_camera.width, old_camera.height, old_camera.params)
      new_loc_cameras[new_camera.id] = new_camera

    # 修改image，只修改id，其他不变
    # image的id是不会重复的，所以直接添加
    new_image = Image(image_id_in_locmap, image.qvec, image.tvec, camera_id_in_locmap, image.name, image.xys, image.point3D_ids)
    new_loc_images[new_image.id] = new_image
  assert(len(loc_cameras) == len(new_loc_cameras))
  assert(len(loc_images) == len(new_loc_images))
  print('modify points3d...')
  # 循环loc-model的points3d
  for _, point3d in loc_points3d.items():
    # 修改points3d，只修改它对应的image_ids，其他不变
    new_image_ids = point3d.image_ids
    for idx in range(len(point3d.image_ids)):
      image_id = point3d.image_ids[idx]
      image_name = loc_images[image_id].name
      image_id_in_locmap = locmap_name_id_dict[image_name][0]
      new_image_ids[idx] = image_id_in_locmap
    new_point3d = Point3D(point3d.id, point3d.xyz, point3d.rgb, point3d.error, new_image_ids, point3d.point2D_idxs)
    new_loc_points3d[new_point3d.id] = new_point3d
  assert(len(loc_points3d) == len(new_loc_points3d))
  print('write new model...')
  # 写新的model
  write_model(new_loc_cameras, new_loc_images, new_loc_points3d, new_loc_model_path, '.bin')
  return

def modify_image_model_binary(old_model_path, database_path, new_model_path, write_ext = '.txt'):
  if os.path.exists(old_model_path + "/images.bin"):
    images_model = read_images_binary(old_model_path + "/images.bin")
  elif os.path.exists(old_model_path + "/images.txt"):
    images_model = read_images_text(old_model_path + "/images.txt")
  else:
    print('failed to read ', old_model_path + "/images.txt or bin")
  # id=image_id, qvec=qvec, tvec=tvec,
  #               camera_id=camera_id, name=image_name,
  #               xys=xys, point3D_ids=point3D_ids)
  # for _, image in image_model.items():
  #   print('image_id ', image.id, ', camera_id ', image.camera_id, ', image_name ', image.name)

  db = COLMAPDatabase.connect(database_path)

  # For convenience, try creating all the tables upfront.

  db.create_tables()
  # camera_id, model, width, height, params, prior = next(rows)


  cursor = db.execute("SELECT image_id, camera_id, name FROM images;")

  images_db = {}

  set_camera = set([])

  for row in cursor:
    image_id = row[0]
    camera_id = row[1]
    image_name = row[2]
    qvec=np.zeros(4)      #pose留空
    tvec=np.zeros(3)
    xys = []
    point3D_ids = []

    for _, image in images_model.items():
      if image.name==image_name:
        qvec=image.qvec
        tvec=image.tvec
        set_camera.add((camera_id, image.camera_id))
  #   print('image_id ', image.id, ', camera_id ', imagcamerase.camera_id, ', image_name ', image.name)


        images_db[image_id] = Image(
          id=image_id, qvec=qvec, tvec=tvec,
          camera_id=camera_id, name=image_name,
          xys=xys, point3D_ids=point3D_ids)
        break

  folder = os.path.exists(new_model_path)
  if not folder:
    os.makedirs(new_model_path)

  print("set_camera", set_camera)

 # write_images_text(images_db, new_model_path + "/images.txt")
  

#  file = open(new_model_path + "/points3D.txt",'w')
#  file.close()


  cameras_db = read_cam_model_from_db(database_path)
  if os.path.exists(old_model_path + "/cameras.bin"):
    cameras_model = read_cameras_binary(old_model_path + "/cameras.bin")
  elif os.path.exists(old_model_path + "/cameras.txt"):
    cameras_model = read_cameras_text(old_model_path + "/cameras.txt")
  else:
    print('failed to read ', old_model_path + "/cameras.txt or bin")

  for item in set_camera:
    tmp_id =  cameras_db[item[0]].id
   # cameras_db[item[0]] = cameras_model[item[1]]
    cameras_db[item[0]] = Camera(id=tmp_id,
                                model=cameras_model[item[1]].model,
                                width=cameras_model[item[1]].width,
                                height=cameras_model[item[1]].height,
                                params=cameras_model[item[1]].params)

  #write_cameras_text(cameras_db, new_model_path + "/cameras.txt")

  points3D = {}
  write_geo_text(images_db, new_model_path + "/geos.txt")
  write_model(cameras_db, images_db, points3D, new_model_path, write_ext)
  

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_file', required=True)
    parser.add_argument('--input_model', required=True)
    parser.add_argument('--output_model', required=True)
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    update_loc_model_id_refer_to_locmap_database(args.input_model, args.database_file, args.output_model)
    # modify_image_model_binary(args.input_model, args.database_file, args.output_model)


if __name__ == '__main__':
    main()

# modify_image_model_binary(
#   '/media/netease/Software/Dataset/LandmarkAR/DistrictC/C11_F2_inv30/sparse/imagereg/images.bin',  
#   '/media/netease/Software/Dataset/LandmarkAR/DistrictC/C11_F2_inv30/database_brief.db',
#   '/media/netease/Software/Dataset/LandmarkAR/DistrictC/C11_F2_inv30/sparse/brief_pre/', 
#   ) 