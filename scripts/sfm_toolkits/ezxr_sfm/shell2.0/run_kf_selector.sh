#######################################################################################
# 请配置以下参数
# SCENE_GRAPH_NAME 直接复制文件名（不含后缀和路径）即可，前提是该json文件保存在PROJECT_PATH下
# IMAGE_LIST_SUFFIX是videos解析时生成的image_list的后缀

COLMAP_FOLDER_PATH=/home/lzx/lzx-codes/colmap_sfm
PROJECT_PATH=/media/lzx/软件/lzx-data/qj_city_block_a
SCENE_GRAPH_NAME='scene_graph_base'
IMAGE_LIST_SUFFIX=''
#######################################################################################

cd $COLMAP_FOLDER_PATH/scripts/sfm_toolkits/ezxr_sfm/colmap_process_loop_folders/

if [ -z "$IMAGE_LIST_SUFFIX_TRY" ]; then

python colmap_kfs_selecter.py \
--projPath $PROJECT_PATH \
--sceneGraph $SCENE_GRAPH_NAME

else

python colmap_kfs_selecter.py \
--projPath $PROJECT_PATH \
--sceneGraph $SCENE_GRAPH_NAME \
--imageListSuffix $IMAGE_LIST_SUFFIX

fi

