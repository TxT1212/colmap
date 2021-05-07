# python3 run_extra_imgreg.py \
# --colmap_folder /media/administrator/dataset/oasis/C6/C6_part_F1Hall \
# --base_db database.db \
# --new_db imagereg.db \
# --input_model sparse/base \
# --output_model sparse/imagereg \
# --image_type video \
# --voc_file /home/administrator/git/colmap/vocabulary/vocab_tree_flickr100K_words256K.bin   \
# --check \
#--jump #creator feature match mapper aligner


# python3 run_extra_imgreg.py \
# --colmap_folder /media/administrator/dataset/oasis/LandmarkAR/C11_mobile_subtests/C11_zhangyan_testmetrics_F2 \
# --base_db database_sfm_geo.db \
# --new_db imagereg.db \
# --input_model sparse/geo \
# --output_model sparse/imagereg \
# --image_type video \
# --voc_file /home/administrator/git/colmap/vocabulary/vocab_tree_flickr100K_words256K.bin   \
# --check \

python3 run_extra_imgreg.py \
--colmap_folder /media/administrator/dataset/LargeSceneAR/TomJerry_Stair/colmap_model/C6_part_FNStairS \
--base_db database.db \
--new_db imagereg.db \
--input_model sparse/imagereg \
--output_model sparse/loc \
--image_type individual \
--voc_file /home/administrator/git/colmap/vocabulary/vocab_tree_flickr100K_words256K.bin   \
# --check \