# 额外图像注册使用方法
见 `run_extra_imgreg_example.sh`

预处理：将sfm骨架复制一份，将需要注册的图像放入proj/images/append文件夹下
执行以下命令

```
python3 run_extra_imgreg.py \
--colmap_folder $PROJECT \    项目位置
--base_db database_sfm_geo.db \    骨架对应的db
--new_db imagereg.db \           额外图像注册后，对应的db
--input_model sparse/geo \          骨架对应的model（真实尺度）
--output_model sparse/imagereg \        额外图像注册后，输出的model位置
--image_type individual \             图像类型  video 或 individual  关系到匹配策略不同
--voc_file /home/administrator/git/colmap/vocabulary/vocab_tree_flickr100K_words256K.bin   \  vocfile位置
```

执行后 输出model在sparse/imagereg  输出对应的db为 imagereg.db

# 新feature获得三角化点云使用方法
hfnet 自定义脚本参考 https://tower.im/teams/793823/tencent_documents/29935/
自动化脚本：run_hfnet_map_build_example.sh

预处理： 先按`额外图像注册使用方法` 获得注册后的db和model；
运行hfnet特征提取， 将.jpg.txt，.npz文件放在features文件夹下使得相对路径和images下图片对应。

例：图像为 images/append/seq1/1.jpg，   
对应特征文件应该在:    
features/append/seq1/1.jpg.txt   
features/append/seq1/1.npz   
```
python3 run_hfnet_map_build.py \
--colmap_folder $PROJECT \    项目位置
--base_db imagereg.db \    额外图像注册后，siftdb
--new_db database_hfnetcpp_new.db \    新的对应特征的db的名称
--input_model sparse/imagereg \      额外图像注册后，sift-model位置
--output_model sparse/hfnetcpp_new    输出的model的路径 建议和db同名
```


# colmap重建使用外部信息
- 标定过的相机
- 外接特征提取
- 外接特征匹配
- 外接已有gt

1. 标定过的相机 
如果有大畸变，则建议在特征提取之后，特征匹配之前手动将camera信息输入到db文件中，使特征匹配可以利用camera信息。
否则直接在手动创建model的过程中，将相机参数输入即可

2. 外接已有gt
在服务器上使用特征提取时，可能出现图像id和特征id不一致的情况。建议以gt文件中的图像名-图像id-相机id为准。修改db中的图像id信息



# 手机图像采集数据处理

## 数据存放
每次采集的训练数据按以下方式放置
```
/path/to/数据集
+── 场景名_日期_设备_画面方向_(天气_对焦)  
│   +── videos 手机导出的视频
    │   +── xxxx1.mp4
    │   +── xxxx2.mp4
    │   +── ...
    │   +── xxxxN.mp4
│   +── 场景名_日期_设备_画面方向_(天气_对焦)_取帧间隔 视频转出的图像  
```

文件夹命名样例：
C11_mobile_dataset/C11_0305_mate20_vertical_sunny_auto/videos  

室外场景需标明天气


## 数据转图像
建议间隔为10帧取一帧，分辨率为1280*720
```
python3 export_image_from_video.py \
	--video_path /PATH/C11_mobile_dataset/C11_0305_mate20_vertical_sunny_auto/videos  \
	--video_ext ".mp4"  \
	--output_path /PATH/C11_mobile_dataset/C11_0305_mate20_vertical_sunny_auto/C11_0305_mate20_vertical_sunny_auto_1280_inv10 \
    --interval 10 \
    --resize \
	--width 1280 \
	--height 720 
```
`run_video2image.sh`中有具体示例

## colmap图像配准
Colmap SFM的文件结构
```
/path/to/project/场景
+── images
│   +── raw_sfm1(放置单反/手机拍摄的高精度图像)
│   +── ...
│   +── raw_sfmN
    │   +── image1.jpg
    │   +── ...
    │   +── imageN.jpg
    +── mobile_image1(待注册到已有模型的手机拍摄图像文件夹)
    │   +── image1.jpg
    │   +── image2.jpg
    │   +── ...
    │   +── imageN.jpg
    +── mobile_image2
+── database.db
+── sparse
│   +── geo(放置单反/手机拍摄的高精度图像)
```

1. 分批次将一批数据拷贝到 colmapSFM场景/images下面，建议每批数据在10000张以内，按视频序列来
2. 运行run_extra_imgreg.sh, 配准后的模型存在场景/sparse/imagereg下面
3. 将模型拷贝到数据集对应的下路径，命名为
```
/path/to/数据集
+── 场景名_日期_设备_画面方向_(天气_对焦)  
│   +── 场景名_日期_设备_画面方向_(天气_对焦)_取帧间隔_model   
```
4. 删除步骤1中拷贝到Colmap SFM 文件结构中的图像数据，删除生成的场景/sparse/imagereg模型
5. 换一批数据重复1-4

注：
在`run_extra_imgreg.sh`中最后将二进制model转换为txt时，有poseonly的选项可选，默认开启，则只输出pose。如果关闭，则会输出稀疏的2D-3D匹配等信息

