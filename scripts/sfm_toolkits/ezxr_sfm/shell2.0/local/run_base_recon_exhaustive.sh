#######################################################################################
# 请配置以下参数
# SCENE_GRAPH_NAME 直接复制文件名（不含后缀和路径）即可，前提是该json文件保存在PROJECT_PATH下

COLMAP_FOLDER_PATH=/home/mm/ARWorkspace/colmap/
PROJECT_PATH=/data/largescene/ezxr_bd/wh-gzsct/
SCENE_GRAPH_NAME='scene_graph_base'

#######################################################################################


cd $COLMAP_FOLDER_PATH/scripts/sfm_toolkits/ezxr_sfm/colmap_process_loop_folders/

# python base_reconstruction.py \
# --projPath $PROJECT_PATH \
# --sceneGraph $SCENE_GRAPH_NAME \
# --useExhaustiveSFM \
# --boardParam $COLMAP_FOLDER_PATH/scripts/sfm_toolkits/ezxr_sfm/python_scale_ezxr/config/board_ChArUco_24x12.yaml

# ########################
# ##如果model的轴向不对, 运行如下功能
# # 请配置以下参数

COLMAP_MODEL_PATH=/data/largescene/ezxr_bd/wh-gzsct/sparse/wh-gzsctwh_CharucoDeleted/0
########################
cd $COLMAP_FOLDER_PATH/scripts/sfm_toolkits/ezxr_sfm/colmap_process
python colmap_export_geo.py \
--model $COLMAP_MODEL_PATH \
--orientation 0 0 -1 0 1 0 1 0 0
#--orientation 0 0 1 0 1 0 -1 0 0 

colmap model_aligner \
--input_path $COLMAP_MODEL_PATH \
--ref_images_path $COLMAP_MODEL_PATH/geos.txt \
--output_path $COLMAP_MODEL_PATH \
--robust_alignment_max_error 0.05 
