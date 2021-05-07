# coding: utf-8
import logging

#日志系统，既要把日志输出到控制台，还要写入日志文件
'''
https://www.cnblogs.com/goodhacker/p/3355660.html

用法: logger = Logger(logfilename='log.txt', logger="fox").getlog()
说明：
logfilename:    日志写入到该文件
logger:         是整个python运行过程中，全局变量key(唯一性)
'''
class Logger():
    def __init__(self, logfilename, logger):
        # 创建一个logger
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.INFO)

        # 创建一个handler，用于写入日志文件;md,不知道为什么写入日志的信息也会打印到控制台
        fh = logging.FileHandler(logfilename)
        fh.setLevel(logging.INFO)

        # 再创建一个handler，用于输出到控制台
        # ch = logging.StreamHandler()
        # ch.setLevel(logging.INFO)

        # 定义handler的输出格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        # ch.setFormatter(formatter)

        # 给logger添加handler
        self.logger.addHandler(fh)
        # self.logger.addHandler(ch)

    def getlog(self):
        return self.logger