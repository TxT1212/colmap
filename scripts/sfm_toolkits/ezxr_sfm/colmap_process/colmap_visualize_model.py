import sys
import sqlite3
import math
import numpy
import cv2
import os
from tqdm import tqdm

sys.path.append('../')
from colmap_process.colmap_read_write_model import *
from colmap_process.colmap_db_parser import *

def distortion_simple_radial(k, u, v):
    radial = k * (u * u + v * v)
    du = u * radial
    dv = v * radial
    return du, dv

def world2image_simple_radial(params, u, v):
    if (len(params) != 4):
        strs = "Error! len(params) = " + str(len(params))
        raise ValueError(strs)
    f = params[0]
    cx = params[1]
    cy = params[2]
    k = params[3]
    du, dv = distortion_simple_radial(k, u, v)
    x = u + du
    y = v + dv
    x = f * x + cx
    y = f * y + cy
    return x, y

def world2image(params, u, v):
    fx = 0
    fy = 0
    cx = 0
    cy = 0
    if len(params) == 3:
        fx = params[0]
        fy = fx
        cx = params[1]
        cy = params[2]
    elif len(params) >= 4:
        fx = params[0]
        fy = params[1]
        cx = params[2]
        cy = params[3]

    x = fx * u + cx
    y = fy * v + cy
    return x, y

def T_w2i_to_T_i2w(img):
    r = quaternion_matrix(img.qvec)[0:3, 0:3]
    rmat = r.transpose()
    tvec = img.tvec
    tnew = -rmat @ tvec
    return rmat, tnew

def pt_global_to_camera(img, xyz):
    # img的pose是from world to image
    pt3d = np.matmul(img.qvec2rotmat(), xyz.reshape(3, 1)) + img.tvec.reshape(3, 1)
    if pt3d[2] < 0.1 : # 小于0是在相机后面, 接近0也不行, 都返回false
        return False, 0.0, 0.0
    return True, pt3d[0] / pt3d[2], pt3d[1] / pt3d[2]

def colmap_visual(model_path, image_path, output_folder, interval = 100, image_name = ""):
    cameras, images, points3D = read_model(model_path, ".bin")

    single_image = False
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    if image_name != "":
        single_image = True
        
    print("Finish reading...")
    camid = -1
    params = {}
    img_counter = 0

    
   
    for _, image in images.items():
        img_counter = img_counter + 1
       # print(image.name)
        if single_image and image.name != image_name:
            continue
        
        if not single_image and img_counter % interval != 0 :
            continue 

        image_file = cv2.imread(image_path + '/' + image.name)
        height, width, channels = image_file.shape

        camid = image.camera_id
        qvec=image.qvec
        tvec=image.tvec

        for _, cam in cameras.items():
            if cam.id == camid:
                camid = image.camera_id
                params = cam.params
                if cam.model!="SIMPLE_RADIAL":
                    continue

                for _, pt in tqdm(points3D.items()):
                    pt3d_src = pt.xyz.reshape(3, 1)
                    
                    ret, u, v = pt_global_to_camera(image, pt3d_src)
                    
                    if ret == True:
                        #   x,y = world2image(params, u, v)
                        x, y = world2image_simple_radial(params, u, v) 
                        #print(pt3d_src,(u,v),(x,y))
                        if x <= 0 or x >= width or y <= 0 or y >= height:
                            continue

                        cv2.circle(image_file, (x,y),3, (int(pt.rgb[2]), int(pt.rgb[1]),int(pt.rgb[0])),-1)

                cv2.resize(image_file, (1920, 1080))
              #  cv2.imshow("QAQ", image_file)
                father_path = output_folder + "/" + os.path.split(image.name)[0] 
                if not os.path.isdir(father_path):
                    os.makedirs(father_path)
              #  print(output_folder + "/" + image.name)
                print(output_folder + "/" + image.name)
                cv2.imwrite(output_folder + "/" + image.name, image_file)
           #     cv2.waitKey(0)
           
        

def main():
    colmap_visual("/data/largescene/qj_city_block_a/sparse/blockA/0" , \
        "/data/largescene/qj_city_block_a/images/",\
            "/data/largescene/qj_city_block_a/preview/", \
                20
            )

if __name__ == '__main__':
    main()   