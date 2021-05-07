# coding: utf-8
import os
import sys
import math
import numpy as np

# -----simple_radial-----
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
#-------------------------