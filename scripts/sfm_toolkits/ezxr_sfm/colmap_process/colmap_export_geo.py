# coding: utf-8
import sys
import sqlite3
import math
import numpy

sys.path.append('../')
from colmap_process.colmap_read_write_model import *
from colmap_process.colmap_db_parser import *


#from scipy.spatial.transform import Rotation as R

def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> numpy.allclose(quaternion_from_matrix(R, isprecise=False),
    ...                quaternion_from_matrix(R, isprecise=True))
    True

    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:4, :4]
    if isprecise:
        q = numpy.empty((4, ))
        t = numpy.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = numpy.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = numpy.linalg.eigh(K)
        q = V[[3, 0, 1, 2], numpy.argmax(w)]
    if q[0] < 0.0:
        numpy.negative(q, q)
    return q

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
  q = numpy.array(quaternion, dtype=numpy.float64, copy=True)
  n = numpy.dot(q, q)
  if n < 1e-9:
      return numpy.identity(3)
  q *= math.sqrt(2.0 / n)
  q = numpy.outer(q, q)
  return numpy.array([
      [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0]],
      [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0]],
      [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]]
      ])

def write_geo_text(images, path, orientation):
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
      ori = [
        [int(orientation[0]), int(orientation[1]), int(orientation[2])],
        [int(orientation[3]), int(orientation[4]), int(orientation[5])],
        [int(orientation[6]), int(orientation[7]), int(orientation[8])]
      ]
      tnew = ori @ tnew
    #   tnew[0] = tnew[0] * int(orientation[0])
    #   tnew[1] = tnew[1] * int(orientation[1])
    #   tnew[2] = tnew[2] * int(orientation[2])

      image_header = [img.name, *tnew]
      first_line = " ".join(map(str, image_header))
      fid.write(first_line + "\n")

def colmap_export_geo(model_path, orientation):
    images_model = read_images_binary(model_path+ "/images.bin")
    write_geo_text(images_model, model_path + "/geos.txt", orientation)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--orientation', type=str, nargs='+', default=[1,0,0,0,1,0,0,0,1])
    parser.add_argument('--output', default="geos.txt")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if (len( args.orientation)) != 9:
        print("invalid orientation rotation matrix")
        return

    if os.path.exists(args.model+ "/images.bin"):
    	images_model = read_images_binary(args.model+ "/images.bin")
    elif os.path.exists(args.model+ "/images.txt"):
    	images_model = read_images_text(args.model+ "/images.txt")
    else:
        print("images.bin or images.txt not exists")
        return

    write_geo_text(images_model, args.model + "/" + args.output, args.orientation)

if __name__ == '__main__':
    main()

