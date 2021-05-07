SCRIPT_PATH="D:/ARWorkspace/colmap/scripts/sfm_toolkits"

cd "$SCRIPT_PATH/pipeline"

pwd

python raw_data_prepare.py \
    --video_path E:/LargeScene/Scene/XixiWetland/raw_cam/butterflyyard_oner_0812_sunny/  \
    --colmap_path   E:/LargeScene/Scene/XixiWetland/colmap_model/butterflyyard  \
    --script_path   $SCRIPT_PATH \
    --video_multithread 1 \
    --jump video