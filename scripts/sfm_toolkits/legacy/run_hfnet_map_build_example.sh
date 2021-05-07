python3 run_hfnet_map_build.py \
--colmap_folder "/media/administrator/dataset/LargeSceneAR/TomJerry_Stair/colmap_model/C6_part_FNStairS/" \
--base_db imagereg.db \
--new_db database_hfnetcpp_testcam.db \
--input_model sparse/loc \
--output_model sparse/hfnetcpp \
--jump external_match \
--check