#######################################################################################
# 请配置以下参数
COLMAP_FOLDER_PATH=/home/mm/ARWorkspace/colmap/
PROJECT_PATH=/data/largescene/C11_base/
SRC_PATH=$PROJECT_PATH/videos
DST_PATH=$PROJECT_PATH/images
SMALL_DST_PATH=$PROJECT_PATH/images_ds
INTERVAL=15
SHORT_SIZE=640
IMAGE_LIST_SUFFIX=''
MULTITHREAD=true
#######################################################################################


cd $COLMAP_FOLDER_PATH/scripts/sfm_toolkits/ezxr_sfm/colmap_process_loop_folders/

if $MULTITHREAD; then
   if [ -z "$IMAGE_LIST_SUFFIX" ]; then
      python videos_to_images_with_lists.py \
      --srcpath $SRC_PATH \
      --dstpath $DST_PATH \
      --smalldstpath $SMALL_DST_PATH \
      --interval $INTERVAL \
      --shortsize $SHORT_SIZE \
      --imageListSuffix '' \
      --multithread 
   else
      python videos_to_images_with_lists.py \
      --srcpath $SRC_PATH \
      --dstpath $DST_PATH \
      --smalldstpath $SMALL_DST_PATH \
      --interval $INTERVAL \
      --shortsize $SHORT_SIZE \
      --imageListSuffix $IMAGE_LIST_SUFFIX \
      --multithread 
   fi
else
   if [ -z "$IMAGE_LIST_SUFFIX" ]; then
      python videos_to_images_with_lists.py \
      --srcpath $SRC_PATH \
      --dstpath $DST_PATH \
      --smalldstpath $SMALL_DST_PATH \
      --interval $INTERVAL \
      --shortsize $SHORT_SIZE \
      --imageListSuffix ''
   else
      python videos_to_images_with_lists.py \
      --srcpath $SRC_PATH \
      --dstpath $DST_PATH \
      --smalldstpath $SMALL_DST_PATH \
      --interval $INTERVAL \
      --shortsize $SHORT_SIZE \
      --imageListSuffix $IMAGE_LIST_SUFFIX 
   fi
fi



