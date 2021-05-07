import collections
import numpy as np
# a = np.zeros(2)
# b = np.ones(2)
a = [0, 0]
# b = [1, 1]
# ab = np.column_stack(tuple([a, b]))
# ab = (tuple([a, b]))
# print(ab)
# print(a)


arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
a = arr.dot(np.array([[1, 0, 0]]).T) 

print(a)
print(a.T)
# print(arr[1])

# Point = collections.namedtuple('Point', ['x', 'y'])
# p = Point(x="[]", y=22)
# p = p._replace(x=221)
# print(p.x)

# aa = []

# aa.insert(2, 3)
# aa.append(1)
# # aa[3] = 3
# print(aa)

# images_points_dictionary = []
# for i in range(3):
#     images_points_dictionary.append(i)
# print(np.stack(images_points_dictionary, axis=0))
# print(type(np.stack(images_points_dictionary, axis=0)))
# print(images_points_dictionary)
# tup2 = (1, 2, 3, 4, 5, 6)
# x = tuple(map(int, tup2[0::2]))
# y = tuple(map(int, tup2[1::2]))
# print(x)
# print(y)
# xy = np.column_stack([x, y])
# print(type(xy))
# print(np.column_stack([x, y]))
# print(xy.reshape([2, -1]))

# rgb = np.ones(3, int)
# print(rgb)
# rgb[0]*=-1
# print(rgb)
a = np.array([[ 9.76081567e-01,-1.74519254e-02,-2.16703032e-01, 1.11298845e+00],
 [1.81277339e-02, 9.99835039e-01, 1.13104252e-03, 1.30528854e-01],
 [ 2.16647546e-01, -5.03232465e-03,  9.76236916e-01,  2.16829222e+00],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
a= a[0:3, 0:3]
a.dot(a.T)
print(a)
print(np.linalg.norm([1, 1, 1]))