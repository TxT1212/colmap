import numpy as np
import argparse
from read_write_model import *
import math


def calculate_error(images, images_gt):
    distances = []
    angles = []
    for _, img in images.items():
        name = img.name
        qvec = img.qvec
        rvc = qvec2rotmat(qvec)
        tvec = img.tvec
        tvec = -(rvc.T).dot(tvec)
        

        # print("T\n", T_img)
        for _, img_gt in images_gt.items():
            if img_gt.name == name:
                qvec_gt = img_gt.qvec
                rvec_gt = qvec2rotmat(qvec_gt)
                tvec_gt = img_gt.tvec
                tvec_gt = -(rvec_gt.T).dot(tvec_gt)
                # print("gt\n", rvec_gt, tvec_gt)
                distance = np.linalg.norm(tvec_gt - tvec.squeeze())
                rotation = rvc.dot(rvec_gt.T)
                # print("test: T*T': \n", rvc.dot(rvec_gt.T))
                angle = (rotation.dot(np.array([[0, 0, 1]]).T))[2][0]
                angle = round(angle, 10)
                # print("angle ", angle)
                if angle > 1:
                    angle =1
                angle = math.acos(angle)
                distances.append(distance)
                angles.append(angle)
                # print(angle, distance)
                break
    return np.array(angles), np.array(distances)

def main1():
    parser = argparse.ArgumentParser(
        description="Read COLMAP model and caluate difference with ground truth")
    parser.add_argument("--input_model", help="path to input model folder")
    parser.add_argument("--input_format", choices=[".bin", ".txt"],
                        help="input model format", default="")

    parser.add_argument("--input_model_gt", help="path to input model folder")
    parser.add_argument("--input_format_gt", choices=[".bin", ".txt"],
                        help="input ground truth model format", default="")

    parser.add_argument("--output_result", help="path to result output file")

    args = parser.parse_args()

    cameras_gt, images_gt, points3D_gt = read_model(
        path=args.input_model_gt, ext=args.input_format_gt)
    cameras, images, points3D = read_model(
        path=args.input_model, ext=args.input_format)

    # print("num_cameras:", len(cameras))
    # print("num_images:", len(images))
    # print("num_points3D:", len(points3D))
    # print("num_cameras gt:", len(cameras_gt))
    # print("num_images gt:", len(images_gt))
    # print("num_points3D gt:", len(points3D_gt))

    angles, distances = calculate_error(images, images_gt)
    # All conditions: (0.25m, 2°) / (0.5m, 5°) / (5m, 10°)
    cout_excelent = 0
    cout_good = 0
    cout_ok = 0
    # print(angles.size)
    for i in range(angles.size):
        if (distances[i] < 0.25 and angles[i] < 2):
            cout_excelent+=1 
        if (distances[i] < 0.5 and angles[i] < 5):
            cout_good+=1 
        if (distances[i] < 5 and angles[i] < 10):
            cout_ok+=1 
    print("0.25m, 2°:",cout_excelent, "/", angles.size, " ", cout_excelent*1.0/angles.size)
    print("0.5m, 5°:",cout_good, "/", angles.size, " ",  cout_good*1.0/angles.size)
    print("5m, 10°:",cout_ok, "/", angles.size, " ",  cout_ok*1.0/angles.size)
if __name__ == "__main__":
    main1()
