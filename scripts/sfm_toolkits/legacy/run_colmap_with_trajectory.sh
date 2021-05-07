PROJECT_PATH=/media/administrator/PoorMan/cartographer_data/lidar_2020-04-02-11-24-39
colmap feature_extractor \
    --database_path $PROJECT_PATH/database.db \
    --image_path $PROJECT_PATH/images

colmap exhaustive_matcher
    --database_path $PROJECT_PATH/database.db