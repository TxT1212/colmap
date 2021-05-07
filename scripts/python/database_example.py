# coding: utf-8
import os
import numpy as np
from database import *
from read_write_model import *
import cv2
import pcl

def get_default_camera_model(img_path_name, camera_id):
    img = cv2.imread(img_path_name)
    if img is None:
        print('failed to load image:', img_path_name)
        return None
    # 行, 列, 通道
    hh, ww, cc = img.shape
    focal_length = ww * 1.2
    cx = ww * 0.5
    cy = hh * 0.5
    elems = [focal_length, cx, cy, 0.0]
    params = np.array(tuple(map(float, elems)))
    cam = Camera(id=int(camera_id), model='SIMPLE_RADIAL', width=ww, height=hh, params=params)
    return cam

def get_fisheye_camera_model(img_path_name, camera_id):
    img = cv2.imread(img_path_name)
    if img is None:
        print('failed to load image:', img_path_name)
        return None
    # 行, 列, 通道
    hh, ww, cc = img.shape
    focal_length = ww * 1.2
    cx = ww * 0.5
    cy = hh * 0.5
    elems = [focal_length, cx, cy, 0.0, 0.0]
    params = np.array(tuple(map(float, elems)))
    cam = Camera(id=int(camera_id), model='RADIAL_FISHEYE', width=ww, height=hh, params=params)
    return cam

def get_image_model(tum, image_id, camera_id, image_name):
    # tum是TC2W,但是colmap需要TW2C
    # tum_inv = io_tool.tum_str_inv(tum)
    tum_inv = tum
    colmap_tvec = [tum_inv[1], tum_inv[2], tum_inv[3]]
    tvec = np.array(tuple(map(float, colmap_tvec)))
    colmap_qvec = [tum_inv[7], tum_inv[4], tum_inv[5], tum_inv[6]]
    qvec = np.array(tuple(map(float, colmap_qvec)))
    xys = np.array([''])
    point3D_ids = np.array([''])
    img = Image(id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=image_name, xys=xys, point3D_ids=point3D_ids)
    return img



if __name__ == '__main__':
    db = COLMAPDatabase.connect(database_path_name)
    db.create_tables()

    cam = get_default_camera_model(first_image, 1)
    model_id = CAMERA_MODEL_NAMES[cam.model].model_id
    db.add_camera(model_id, cam.width, cam.height, cam.params, prior_focal_length=False, camera_id=cam.id)
    img = get_image_model(traj, count, cam.id, "cam/" + time_str + ".png")
    db.add_image(img.name, img.camera_id, prior_q=img.qvec, prior_t=img.tvec, image_id=img.id)

    cam = get_fisheye_camera_model(first_image,i+2)
    model_id = CAMERA_MODEL_NAMES[cam.model].model_id
    db.add_camera(model_id, cam.width, cam.height, cam.params, prior_focal_length=False, camera_id=cam.id)
    db.add_image(img.name, img.camera_id, prior_q=img.qvec, prior_t=img.tvec, image_id=img.id)
    db.commit()

    print('after insert data to table:')
    cursor = db.cursor()
    cursor.execute("select * from cameras")
    results = cursor.fetchall()
    print('cameras number:', len(results))

    cursor = db.cursor()
    cursor.execute("select * from images")
    results = cursor.fetchall()
    print('images number:', len(results))

    cursor = db.cursor()
    cursor.execute("select * from keypoints")
    results = cursor.fetchall()
    print('keypoints number:', len(results))

    db.close()
    print("Raw Data to database success!")