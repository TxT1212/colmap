#######################################################################################
# 请配置以下参数
# SCENE_GRAPH_NAME 直接复制文件名（不含后缀和路径）即可，前提是该json文件保存在PROJECT_PATH下
# 如果将USE_PARTIAL_EXISTING_MODEL设为true,新增数据的mapper会以部分的、与新数据存在关联的现有地图model为输入

COLMAP_FOLDER_PATH=/home/lzx/lzx-codes/colmap_sfm
PROJECT_PATH=/media/lzx/软件/lzx-data/qj_city_block_a
SCENE_GRAPH_NAME='scene_graph_base'
USE_PARTIAL_EXISTING_MODEL=false
#######################################################################################


cd $COLMAP_FOLDER_PATH/scripts/sfm_toolkits/ezxr_sfm/colmap_process_loop_folders/

if $USE_PARTIAL_EXISTING_MODEL;then

python base_reconstruction.py \
--projPath $PROJECT_PATH \
--sceneGraph $SCENE_GRAPH_NAME \
--usePartialExistingModel \
--boardParam $COLMAP_FOLDER_PATH/scripts/sfm_toolkits/ezxr_sfm/python_scale_ezxr/config/board_ChArUco_24x12.yaml

else

python base_reconstruction.py \
--projPath $PROJECT_PATH \
--sceneGraph $SCENE_GRAPH_NAME \
--boardParam $COLMAP_FOLDER_PATH/scripts/sfm_toolkits/ezxr_sfm/python_scale_ezxr/config/board_ChArUco_24x12.yaml

fi



