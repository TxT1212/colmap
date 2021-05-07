#输入无尺度的重建模型，输出z轴朝上的
#脚本3，前置依赖1

DATASET_PATH=$1
SFM_SCRIPT_PATH=$2

# colmap model_orientation_aligner \
# 	--image_path $DATASET_PATH/images \
# 	--input_path $DATASET_PATH/sparse/org/0 \
# 	--output_path $DATASET_PATH/sparse/org \
# 	--max_image_size 1024 \

python $SFM_SCRIPT_PATH/colmap_data/colmap_export_geo.py \
  --model $DATASET_PATH/sparse/org \
  --output geosz.txt \
  --orientation 1 0 0 0 0 1 0 -1 0

mkdir $DATASET_PATH/sparse/gravity
colmap model_aligner \
  --input_path $DATASET_PATH/sparse/org \
  --ref_images_path $DATASET_PATH/sparse/org/geosz.txt  \
  --output_path $DATASET_PATH/sparse/gravity \
  --robust_alignment_max_error 0.05

