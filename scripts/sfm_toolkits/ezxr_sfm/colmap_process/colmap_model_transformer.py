# coding: utf-8
import sys
sys.path.append('../')
from colmap_process.colmap_read_write_model import *
from colmap_process.colmap_export_geo import *
import shutil
import numpy

def transform_image_model(image_model, rmat, tmat):

  transform_matrix = numpy.identity(4)
  transform_matrix[0:3,0:3] = rmat
  transform_matrix[0:3, 3] = tmat

  new_image_model = {}

  for _, image in image_model.items():
    qvec = image.qvec
    tvec = image.tvec

    image_matrix = numpy.identity(4)
    image_matrix[0:3,0:3] = quaternion_matrix(qvec)
    image_matrix[0:3, 3] = tvec

    new_image_matrix = image_matrix @ numpy.linalg.inv(transform_matrix)

    new_qvec = quaternion_from_matrix(new_image_matrix)
    new_tvec = image_matrix[0:3, 3]

    new_image_model[image.id] = Image(
      id=image.id, qvec=new_qvec, tvec=new_tvec,
      camera_id=image.camera_id, name=image.name,
      xys=image.xys, point3D_ids=image.point3D_ids)

  return new_image_model

def transform_point_model(point_model, rmat, tmat):
  transform_matrix = numpy.identity(4)
  transform_matrix[0:3,0:3] = rmat
  transform_matrix[0:3, 3] = tmat

  new_point_model = {}

  for _, pt in point_model.items():
    xyz = numpy.array([pt.xyz[0], pt.xyz[1], pt.xyz[2],1])
    #xyz = pt.xyz
    xyz_new = transform_matrix @ xyz

    new_point_model[pt.id] = Point3D(
      id=pt.id, xyz=numpy.array([xyz_new[0],xyz_new[1],xyz_new[2]]), rgb=pt.rgb,
      error=pt.error, image_ids=pt.image_ids, 
      point2D_idxs=pt.point2D_idxs)
 
  return new_point_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model', required=True)
    parser.add_argument('--output_model', required=True)
    parser.add_argument('--rmat', type=str, nargs='+', default=[1,0,0,0,1,0,0,0,1])
    parser.add_argument('--tmat', type=str, nargs='+', default=[0,0,0])
  # parser.add_argument('--output', required=True)
    args = parser.parse_args()
    return args

def transformColmapModel(input_model, output_model, rList=[1,0,0,0,1,0,0,0,1], tList=[0,0,0]):
    rmat = rList
    rotation_matrix = [
        [float(rmat[0]), float(rmat[1]), float(rmat[2])],
        [float(rmat[3]), float(rmat[4]), float(rmat[5])],
        [float(rmat[6]), float(rmat[7]), float(rmat[8])]
    ]
    translation = [float(tList[0]), float(tList[1]), float(tList[2])]

    images_model = read_images_binary(input_model+ "/images.bin")
    new_image_model = transform_image_model(images_model, rotation_matrix, translation)

    point_model = read_points3d_binary(input_model+ "/points3D.bin")
    new_point_model = transform_point_model(point_model, rotation_matrix, translation)

    if not os.path.exists(output_model):
        os.makedirs(output_model)

    write_images_binary(new_image_model, output_model + "/images.bin")
    write_points3d_binary(new_point_model, output_model + "/points3D.bin")

    try:
        shutil.copyfile(input_model + "/cameras.bin", output_model + "/cameras.bin")
    except Exception as e:
        print("warning:")
        print(e)

    return

def main():
    args = parse_args()
    if (len(args.rmat)) != 9:
        print("invalid rotation matrix")
        return
    if (len(args.tmat)) != 3:
        print("invalid translation")
        return   

    rmat = args.rmat
    rotation_matrix = [
        [float(rmat[0]), float(rmat[1]), float(rmat[2])],
        [float(rmat[3]), float(rmat[4]), float(rmat[5])],
        [float(rmat[6]), float(rmat[7]), float(rmat[8])]
    ]
    translation = args.tmat

    images_model = read_images_binary(args.input_model+ "/images.bin")
    new_image_model = transform_image_model(images_model, rotation_matrix, translation)

    point_model = read_points3d_binary(args.input_model+ "/points3D.bin")
    new_point_model = transform_point_model(point_model, rotation_matrix, translation)

    if not os.path.exists(args.output_model):
        os.makedirs(args.output_model)

    write_images_binary(new_image_model, args.output_model + "/images.bin")
    write_points3d_binary(new_point_model, args.output_model + "/points3D.bin")

    try:
        shutil.copyfile( args.input_model + "/cameras.bin", args.output_model + "/cameras.bin")
    except Exception as e:
        print("warning:")
        print(e)

if __name__ == '__main__':
    main()