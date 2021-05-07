## trajectory_tools使用说明

- 环境说明: 
    - 所有脚本在ubuntu16.04下测试通
    - 需使用anaconda的python环境
- 算法说明: 
    - tgt是真值，src对齐到tgt进行误差评估，误差单位是“米”;
    - tgt需要是真实尺度，不然评估得到的结果是没有尺度的，没有物理意义;

评估两个colmap-model/colmap-geos之间的误差:
1. colmap-model/colmap-geos,根据图像名，一一对应地转换到tum格式的txt
```
python colmap_to_tum_evo.py \
--src_path path_to_src_model/geos.txt \
--tgt_path path_to_tgt_model/geos.txt \
--src_output_path path_to_src_evo_format.txt \
--tgt_output_path path_to_tgt_evo_format.txt
```

2. evo工具使用

[请阅读该文档-轨迹分析工具](https://g.hz.netease.com/ARResearch/CalibEva/dataset_collection/-/wikis/%E8%AF%84%E4%BC%B0%E5%B7%A5%E5%85%B7%E5%A4%84%E7%90%86%E6%96%B9%E6%B3%95/%5Bwiki%5D%E8%BD%A8%E8%BF%B9%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7)

注意: evo的所有命令我们都用tum格式，使用之前，请将待评测的对象转换成tum格式

下面给出简单的示例:

2.1 rmse的评估
```
evo_ape tum tgt.txt src.txt -spa
```
- spa代表: 
    - p: plot, 可视化出来
    - s: scale, 做sim3变换，即相似变换
    - a: algin, 对齐，根据一一对应关系，求解transform, 如果不加s,就是刚体变换，如果加s，就是相似变换

2.2 轨迹详细可视化分析(没有误差评估)
```
evo_traj tum --ref tgt.txt src.txt -spa
```

2.3 position精度评估(区别于轨迹，sfm只关注position更合理)
```
python evo_sim3_align_error.py --src_path path_to_src.txt --tgt_path path_to_tgt.txt
```
