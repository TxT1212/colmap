## 合并多个colmap model至一个model，并位移子model中图像相机点的序号。不接受两个model中存在相同图像
import sys
import sqlite3
import math
import numpy as np
import colmap_read_write_model
import colmap_db_parser

from colmap_read_write_model import *
from colmap_db_parser import *



def modify_model(old_img, old_cam, old_pt, img_begin, cam_begin, pt_begin):
  new_img = {}
  new_cam = {}
  new_pt = {}
  for _, image in old_img.items():
    new_imgid = image.id + img_begin
    new_camid = image.camera_id + cam_begin
    new_point3D_ids = image.point3D_ids
    lenvec = new_point3D_ids.shape
    for i in range(lenvec[0]):
      if new_point3D_ids[i] >=0:
        new_point3D_ids[i] = new_point3D_ids[i] + pt_begin

    new_img[new_imgid] = Image(
      id=new_imgid, qvec=image.qvec, tvec=image.tvec,
      camera_id=new_camid, name=image.name,
      xys=image.xys, point3D_ids=new_point3D_ids)

  for _, cam in old_cam.items():
    new_camid = cam.id + cam_begin
    new_cam[new_camid] = Camera(id=new_camid,
                                model=cam.model,
                                width=cam.width,
                                height=cam.height,
                                params=cam.params)
  
  for _, pt in old_pt.items():
    new_pt_id = pt.id + pt_begin

    new_image_ids = pt.image_ids
    lenvec = new_image_ids.shape
    for i in range(lenvec[0]):
      new_image_ids[i] = new_image_ids[i] + img_begin
    
    new_pt[new_pt_id] = Point3D(id=new_pt_id, xyz=pt.xyz, rgb=pt.rgb,
                                               error=pt.error, image_ids=new_image_ids,
                                               point2D_idxs=pt.point2D_idxs)
                                            
  return new_img, new_cam, new_pt

def AddSubModel(submodel_path, mdext, idx, totalcam, totalimg, totalpt):
  cameras, images, points3D = read_model(path=submodel_path, ext=mdext)
  print("cameras",len(cameras))
  print("images",len(images))
  print("points3D",len(points3D))

  img_begin_shift = 0
  cam_begin_shift = 0
  pt_begin_shift = 0

  for _, cam in totalcam.items():
    if cam_begin_shift < cam.id:
      cam_begin_shift = cam.id

  for _, img in totalimg.items():
    if img_begin_shift < img.id:
      img_begin_shift = img.id
  
  for _, pt in totalpt.items():
    if pt_begin_shift < pt.id:
      pt_begin_shift = pt.id

  print("Idx shift: cam ", cam_begin_shift, " img ", img_begin_shift, " pt ", pt_begin_shift)

  new_img, new_cam, new_pt = modify_model(images, cameras, points3D, img_begin_shift, cam_begin_shift, pt_begin_shift)
  totalcam.update(new_cam)
  totalimg.update(new_img)
  totalpt.update(new_pt)

def main():
      #brief hfnetcpp
  models = {
    # "/media/administrator/dataset/oasis/C6/C6_part_F1Hall/sparse/brief",
    # "/media/administrator/dataset/oasis/C6/C6_part_F1N/sparse/brief",
    # "/media/administrator/dataset/oasis/C6/C6_part_F1Ndoor/sparse/brief",
    # "/media/administrator/dataset/oasis/C6/C6_part_F1S/sparse/brief",
    # "/media/administrator/dataset/oasis/C6/C6_part_F2N/sparse/brief",
    # "/media/administrator/dataset/oasis/C6/C6_part_F2S/sparse/brief"
    "/media/lzx/软件/lzx-data/YinXiangJiNan/service/scenes/onlyday/dlreloc/colmap_model",
    "/media/lzx/软件/lzx-data/YinXiangJiNan/dlreloc_onlynight/colmap_model",
  #  "/media/netease/Dataset/LargeScene/Scene/Wanxiangcheng/service/Scene_wxc_f/dlreloc_f2_append/colmap_model",
    # "/media/administrator/dataset/oasis/C6/PartMapModels/LocalizationMap0501/C6_part_F1Hall/sparse/imagereg",
    # "/media/administrator/dataset/oasis/C6/PartMapModels/LocalizationMap0501/C6_part_F1N/sparse/imagereg",
    # "/media/administrator/dataset/oasis/C6/PartMapModels/LocalizationMap0501/C6_part_F1Ndoor/sparse/imagereg",
    # "/media/administrator/dataset/oasis/C6/PartMapModels/LocalizationMap0501/C6_part_F1S/sparse/imagereg",
    # "/media/administrator/dataset/oasis/C6/PartMapModels/LocalizationMap0501/C6_part_F2N/sparse/imagereg",
    # "/media/administrator/dataset/oasis/C6/PartMapModels/LocalizationMap0501/C6_part_F2S/sparse/imagereg"
  }
  extmodel = ".bin"

  output_path = "/media/lzx/软件/lzx-data/YinXiangJiNan/dlreloc_day_night/colmap_model"
  if not os.path.isdir(output_path):
    os.mkdir(output_path)
  imgs={}
  pts={}
  cams={}
  idx = 0
  for path in models:
    print("merge model ", path)
    AddSubModel(path, extmodel, idx, cams, imgs, pts)
    idx = idx+1
  
  write_model(cams, imgs, pts, output_path, ".bin", poseonly=False)
  write_model(cams, imgs, pts, output_path, ".txt", poseonly=False)



if __name__ == '__main__':
    main()