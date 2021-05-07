import colmap_db_parser
import os
import argparse
import sys
import numpy as np

from colmap_db_parser import *

image_gt = '/media/netease/Software/Dataset/Oasis/ibl_mall/sparse/gt_part/images.txt'
cam_gt = '/media/netease/Software/Dataset/Oasis/ibl_mall/sparse/gt_part/cameras.txt'

def import_images(db, imagefile):
  f = open(imagefile, 'r')
  data = f.readlines()
  
  for line in data:
    odom = line.split(' ')    #1 0.313129397762 0.286195975169 -0.610428860533 0.668893452115 166.463034289 12.7322718286 135.937163969 1 query_gt/cdm_20150523_101446.camera
    if len(odom) < 2:
      continue
    image_id = int(odom[0])
    cam_id = int(odom[8])
    image_name = odom[9].strip()
    db.add_image(image_name, cam_id, image_id=image_id)

def import_cameras(db, camfile):
  f = open(camfile, 'r')
  data = f.readlines()
  
  for line in data:
    odom = line.split(' ')   #1 PINHOLE 2064.0 1161.0 1849.562744 1852.409546 1053.160034 608.763672
    cam_id = int(odom[0])
    cam_model = 1
    width = float(odom[2])
    height = float(odom[3])
    params = np.array((float(odom[4]), float(odom[5]), float(odom[6]), float(odom[7].strip())))
    
    db.add_camera(cam_model, width, height, params, camera_id=cam_id)



def create_db_with_gtfile(camfile, imagefile):
  parser = argparse.ArgumentParser()
  parser.add_argument("--database_path", default="database_query_train_par.db")
  args = parser.parse_args()

  ## if os.path.exists(args.database_path):
  #     print("ERROR: database path already exists -- will not modify it.")
  #     return

  # Open the database.

  if os.path.exists(args.database_path):

        print("ERROR: database path already exists -- will not modify it.")
        return

  # Open the database.

  db = COLMAPDatabase.connect(args.database_path)

  # For convenience, try creating all the tables upfront.

  db.create_tables()

  import_images(db, image_gt)
  import_cameras(db, cam_gt)

  db.commit()
  # Create dummy cameras.

  # model1, width1, height1, params1 = \
  #     0, 1024, 768, np.array((1024., 512., 384.))
  # model2, width2, height2, params2 = \
  #     2, 1024, 768, np.array((1024., 512., 384., 0.1))

  # camera_id1 = db.add_camera(model1, width1, height1, params1)
  # camera_id2 = db.add_camera(model2, width2, height2, params2)

  # # Create dummy images.

  # image_id1 = db.add_image("image1.png", camera_id1)
  # image_id2 = db.add_image("image2.png", camera_id1)
  # image_id3 = db.add_image("image3.png", camera_id2)
  # image_id4 = db.add_image("image4.png", camera_id2)

  # # Create dummy keypoints.
  # #
  # # Note that COLMAP supports:
  # #      - 2D keypoints: (x, y)
  # #      - 4D keypoints: (x, y, theta, scale)
  # #      - 6D affine keypoints: (x, y, a_11, a_12, a_21, a_22)

  # num_keypoints = 1000
  # keypoints1 = np.random.rand(num_keypoints, 2) * (width1, height1)
  # keypoints2 = np.random.rand(num_keypoints, 2) * (width1, height1)
  # keypoints3 = np.random.rand(num_keypoints, 2) * (width2, height2)
  # keypoints4 = np.random.rand(num_keypoints, 2) * (width2, height2)

  # db.add_keypoints(image_id1, keypoints1)
  # db.add_keypoints(image_id2, keypoints2)
  # db.add_keypoints(image_id3, keypoints3)
  # db.add_keypoints(image_id4, keypoints4)

  # # Create dummy matches.

  # M = 50
  # matches12 = np.random.randint(num_keypoints, size=(M, 2))
  # matches23 = np.random.randint(num_keypoints, size=(M, 2))
  # matches34 = np.random.randint(num_keypoints, size=(M, 2))

  # db.add_matches(image_id1, image_id2, matches12)
  # db.add_matches(image_id2, image_id3, matches23)
  # db.add_matches(image_id3, image_id4, matches34)

  # # Commit the data to the file.

  # db.commit()

  # # Read and check cameras.

  # rows = db.execute("SELECT * FROM cameras")

  # camera_id, model, width, height, params, prior = next(rows)
  # params = blob_to_array(params, np.float64)
  # assert camera_id == camera_id1
  # assert model == model1 and width == width1 and height == height1
  # assert np.allclose(params, params1)

  # camera_id, model, width, height, params, prior = next(rows)
  # params = blob_to_array(params, np.float64)
  # assert camera_id == camera_id2
  # assert model == model2 and width == width2 and height == height2
  # assert np.allclose(params, params2)

  # # Read and check keypoints.

  # keypoints = dict(
  #     (image_id, blob_to_array(data, np.float32, (-1, 2)))
  #     for image_id, data in db.execute(
  #         "SELECT image_id, data FROM keypoints"))

  # assert np.allclose(keypoints[image_id1], keypoints1)
  # assert np.allclose(keypoints[image_id2], keypoints2)
  # assert np.allclose(keypoints[image_id3], keypoints3)
  # assert np.allclose(keypoints[image_id4], keypoints4)

  # # Read and check matches.

  # pair_ids = [image_ids_to_pair_id(*pair) for pair in
  #             ((image_id1, image_id2),
  #               (image_id2, image_id3),
  #               (image_id3, image_id4))]

  # matches = dict(
  #     (pair_id_to_image_ids(pair_id),
  #       blob_to_array(data, np.uint32, (-1, 2)))
  #     for pair_id, data in db.execute("SELECT pair_id, data FROM matches")
  # )

  # assert np.all(matches[(image_id1, image_id2)] == matches12)
  # assert np.all(matches[(image_id2, image_id3)] == matches23)
  # assert np.all(matches[(image_id3, image_id4)] == matches34)

  # # Clean up.

  db.close()

if __name__ == "__main__":
  create_db_with_gtfile('', '')