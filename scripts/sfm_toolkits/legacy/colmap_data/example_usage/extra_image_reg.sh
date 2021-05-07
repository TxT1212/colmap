#脚本6

DATASET_PATH=$1
SFM_SCRIPT_PATH=$2
VOC_FILE=$3

cd $SFM_SCRIPT_PATH
python $SFM_SCRIPT_PATH/run_extra_imgreg.py \
--colmap_folder $DATASET_PATH \
--base_db database.db \
--new_db imagereg.db \
--input_model sparse/geo \
--output_model sparse/imagereg \
--image_type individual 


## --image_type video \
## --voc_file /home/administrator/git/colmap/vocabulary/vocab_tree_flickr100K_words256K.bin   