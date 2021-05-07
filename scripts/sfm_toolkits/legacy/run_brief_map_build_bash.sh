python3 run_brief_map_build.py \
--colmap_folder /media/administrator/dataset/oasis/C6/C6_part_F1Hall \
--base_db imagereg.db \
--new_db database_brief.db \
--extractor /home/administrator/git/maplab_deros/devel/bin/export_desp_to_file \
--input_model sparse/imagereg \
--output_model sparse/brief \
--check \
--summarymap \
--summap_dir /media/administrator/dataset/oasis/C6/C6_part_F1Hall/summarymap \
--lc_folder /home/administrator/git/maplab_deros/src/localizer/share/lcdat \
--map_builder /home/administrator/git/maplab_deros/devel/bin/test_summary_map_creator 

python3 run_brief_map_build.py \
--colmap_folder /media/administrator/dataset/oasis/C6/C6_part_F2N \
--base_db imagereg.db \
--new_db database_brief.db \
--extractor /home/administrator/git/maplab_deros/devel/bin/export_desp_to_file \
--input_model sparse/imagereg \
--output_model sparse/brief \
--summarymap \
--summap_dir /media/administrator/dataset/oasis/C6/C6_part_F2N/summarymap \
--lc_folder /home/administrator/git/maplab_deros/src/localizer/share/lcdat \
--map_builder /home/administrator/git/maplab_deros/devel/bin/test_summary_map_creator 


#--jump external_feature external_match creator feature reorder match triangulator #mapper aligner


    # parser.add_argument('--summarymap', action='store_true')
    # parser.add_argument('--summap_dir', type=str, default='')
    # parser.add_argument('--lc_folder', type=str, default='')
    # parser.add_argument('--map_builder', type=str, default='')