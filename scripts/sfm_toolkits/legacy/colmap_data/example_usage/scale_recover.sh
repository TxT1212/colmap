#脚本2，前置依赖1

DATASET_PATH=$1
SFM_SCRIPT_PATH=$2

CHARUCO_PATH=$SFM_SCRIPT_PATH/../python_scale_ezxr

cd $CHARUCO_PATH
python run_scale_ezxr.py \
$DATASET_PATH \
$DATASET_PATH/sparse/org/0 

