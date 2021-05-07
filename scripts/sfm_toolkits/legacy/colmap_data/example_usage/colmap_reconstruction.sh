#输入原始图像序列，输出无尺度的重建结果
#脚本1

DATASET_PATH=$1
#camera_model SIMPLE_RADIAL OPENCV_FISHEYE
colmap feature_extractor \
	--database_path $DATASET_PATH/database.db \
	--image_path $DATASET_PATH/images \
	--ImageReader.camera_model OPENCV_FISHEYE  \
	--ImageReader.single_camera_per_folder 1

colmap exhaustive_matcher \
    --database_path $DATASET_PATH/database.db \
    --SiftMatching.min_num_inliers 100

mkdir $DATASET_PATH/sparse
mkdir $DATASET_PATH/sparse/org

colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse/org \
    --Mapper.abs_pose_min_num_inliers 50

