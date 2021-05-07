# coding: utf-8
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_charuco_corners(charuco_corners):
    charuco_pts_gt = []
    for item in charuco_corners.items():
        charuco_pts_gt.append(np.array([item[1][0], item[1][1], 0.0]))
    charuco_pts_gt = np.array(charuco_pts_gt)
    ax = plt.axes(projection="3d")
    ax.scatter(np.array(charuco_pts_gt)[:,0], np.array(charuco_pts_gt)[:,1],
                np.array(charuco_pts_gt)[:,2], c='b')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.axis('equal')
    plt.show()

def visualize_pts_3d(charuco_corners):
    ax = plt.axes(projection="3d")
    ax.scatter(np.array(charuco_corners)[:,0], np.array(charuco_corners)[:,1],
                np.array(charuco_corners)[:,2], c='b')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.axis('equal')
    plt.show()

def visualize_pts_3d_two(group_a, group_b):
    ax = plt.axes(projection="3d")
    ax.scatter(np.array(group_a)[:,0], np.array(group_a)[:,1],
                np.array(group_a)[:,2], c='b')
    ax.scatter(np.array(group_b)[:,0], np.array(group_b)[:,1],
                np.array(group_b)[:,2], c='r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.axis('equal')
    ax.set_aspect('equal', 'box')
    plt.show()