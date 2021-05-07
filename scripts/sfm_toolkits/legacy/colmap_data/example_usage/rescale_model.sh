#脚本4，前置依赖123

DATASET_PATH=$1
SFM_SCRIPT_PATH=$2

python $SFM_SCRIPT_PATH/colmap_data/colmap_model_rescale.py \
  --input_model $DATASET_PATH/sparse/gravity \
  --output_model $DATASET_PATH/sparse/geo \
  --scale 1.6442167579270786   #CJ 1.6442167579270786 0.3704326857364842