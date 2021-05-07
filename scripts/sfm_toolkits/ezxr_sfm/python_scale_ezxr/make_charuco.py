# coding: utf-8
import cv2
from cv2 import aruco
import numpy as np
import sys


def main():
    if len(sys.argv) != 5 and len(sys.argv) != 1:
        print('Generate a ChArUco chessboard with dictionary DICT_4X4_1000.')
        print(
            'make_charuco [board_rows, large than 2] [board_cols, large than 2] [square_length (cm)] [marker_length (cm)]')
        return

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    if len(sys.argv) == 1:
        board_rows = 24
        board_cols = 12
        square_length = 4   # in centimeters
        marker_length = 3   # in centimeters
    else:
        board_rows = int(sys.argv[1])
        board_cols = int(sys.argv[2])
        square_length = float(sys.argv[3])
        marker_length = float(sys.argv[4])
    #入参是x, y,所以是cols和rows
    board = aruco.CharucoBoard_create(board_cols, board_rows, square_length, marker_length, aruco_dict)
    # cv::Size(width, height)
    # opencv里image都是先x后y,mat都是先row后col
    # 即cols和rows
    img = board.draw((200 * board_cols, 200 * board_rows))
    cv2.imwrite('ChArUco_%dx%d.png' % (board_rows, board_cols), img)
    print('Done!')

if __name__ == '__main__':
    main()
