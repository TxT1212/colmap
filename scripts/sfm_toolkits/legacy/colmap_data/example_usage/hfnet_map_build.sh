DATASET_PATH=$1
SFM_SCRIPT_PATH=$2

cd $SFM_SCRIPT_PATH

python $SFM_SCRIPT_PATH/run_hfnet_map_build.py \
--colmap_folder $DATASET_PATH \
--base_db imagereg.db \
--new_db database_hfnetcpp.db \
--input_model sparse/imagereg \
--output_model sparse/hfnetcpp \

