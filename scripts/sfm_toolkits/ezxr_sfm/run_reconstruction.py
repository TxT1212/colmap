# coding: utf-8
import sys
import os
import time
from fileio.parser_config_info import parse_config_file
import pipeline
from pipeline import run_pipeline
from colmap_script.reconstruction import *
from log_file.logstdout import Logstdout


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(sys.argv[0], 'config_file log_path(optional)')
        exit
    config_file = sys.argv[1]

    log_prefix = os.getcwd() + '/log/'
    if len(sys.argv) == 3:
        inputprefix = sys.argv[2]
        if os.path.isabs(inputprefix):
            log_prefix = inputprefix + '/'
        else:
            log_prefix = os.getcwd() + '/log/' + inputprefix + '/'
            
    if not os.path.exists(log_prefix):
        os.makedirs(log_prefix)
    timestring = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
    print(timestring)

    data = parse_config_file(config_file)
    logfilename = log_prefix +  "log_recons_" + data['sceneName'] + "_" + timestring + ".txt"
    
    sys.stdout = Logstdout(logfilename, sys.stdout)
    run_pipeline(data)

    
    