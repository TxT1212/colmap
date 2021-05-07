#######################################################################################
# 请配置以下参数
COLMAP_FOLDER_PATH=/home/mm/ARWorkspace/colmap/
PROJECT_PATH=/data/largescene/xiyou/sceneywm/
JSON_PATH=/data/largescene/xiyou/sceneywm/downloadywm.json
###########################################################################

cd $COLMAP_FOLDER_PATH/scripts/sfm_toolkits/ezxr_sfm/colmap_process_loop_folders/

python download_raw_data.py \
--projPath $PROJECT_PATH \
--routeJson $JSON_PATH 