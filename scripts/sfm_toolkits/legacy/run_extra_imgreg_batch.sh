#0. 运行dataset_collection/sfm_data_toolkit/export_image_from_video.py（可参考同目录下脚本命令），将视频转为给定大小的图像帧
#1. 分批次将一批数据 C11_0305_mate20_vertical_sunny_auto_1280 拷贝到C11/images下面，建议每批数据在10000张以内，按视频序列来
#2. 运行run_extra_imgreg.sh, 模型存在C11/sparse/imagereg下面
#3. 将模型拷贝到目标路径，命名为C11_0305_mate20_vertical_sunny_auto_1280_model
#4. 删除C11/images/C11_0305_mate20_vertical_sunny_auto_1280，   C11/sparse/imagereg
#5. 重复1-4
