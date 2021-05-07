# coding: utf-8
import os
import sys
import argparse
import numpy as np

def read_sim3s_and_write_tum(sim3s_path, tum_path):
    fo = open(sim3s_path, "r")
    fw = open(tum_path, "w")
    # id name x y z q_x q_y q_z q_w s 
    for line in fo.readlines(): # 依次读取每行
        if line[0] == '#':
            continue
        strs = line.split(' ')
        tum_str = strs[0] + ' ' + strs[2] + ' ' + strs[3] + ' ' + strs[4] + ' ' + \
                strs[5] + ' ' + strs[6] + ' ' + strs[7] + ' ' + strs[8] + '\n'
        fw.write(tum_str)
    fo.close()
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim3s_path', required=True)
    parser.add_argument('--tum_path', required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    read_sim3s_and_write_tum(args.sim3s_path, args.tum_path)

if __name__ == "__main__":
    main()
