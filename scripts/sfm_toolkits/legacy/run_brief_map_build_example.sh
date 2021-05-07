# python3 run_brief_map_build.py \
# --colmap_folder /media/netease/Software/Dataset/Oasis/C6/LocMap/LocMap_0512_night/C6_part_F1N \
# --base_db imagereg.db \
# --new_db database_brief.db \
# --extractor /home/netease/ARWorkspace/maplab_deros/devel/bin/export_desp_to_file \
# --input_model sparse/imagereg \
# --output_model sparse/brief \
# --summarymap \
# --summap_dir /media/netease/Software/Dataset/Oasis/C6/LocMap/LocMap_0512_night/C6_part_F1N/summary_map \
# --lc_folder /home/netease/ARWorkspace/maplab_deros/src/localizer/share/lcdat \
# --map_builder /home/netease/ARWorkspace/maplab_deros/devel/bin/test_summary_map_creator  \
# --jump external_feature
# #\
# #--check \
# #--jump external_match #creator feature  match  #mapper aligner reorder triangulator external_feature external_match 
# #--check \


#     # parser.add_argument('--summarymap', action='store_true')
#     # parser.add_argument('--summap_dir', type=str, default='')
#     # parser.add_argument('--lc_folder', type=str, default='')
#     # parser.add_argument('--map_builder', type=str, default='')

# python3 run_brief_map_build.py \
# --colmap_folder /media/netease/Software/Dataset/Oasis/C6/LocMap/LocMap_0512_night/C6_part_F1S \
# --base_db imagereg.db \
# --new_db database_brief.db \
# --extractor /home/netease/ARWorkspace/maplab_deros/devel/bin/export_desp_to_file \
# --input_model sparse/imagereg \
# --output_model sparse/brief \
# --summarymap \
# --summap_dir /media/netease/Software/Dataset/Oasis/C6/LocMap/LocMap_0512_night/C6_part_F1S/summary_map \
# --lc_folder /home/netease/ARWorkspace/maplab_deros/src/localizer/share/lcdat \
# --map_builder /home/netease/ARWorkspace/maplab_deros/devel/bin/test_summary_map_creator  \
# --jump external_feature

# python3 run_brief_map_build.py \
# --colmap_folder /media/netease/Software/Dataset/Oasis/C6/LocMap/LocMap_0512_night/C6_part_F2N \
# --base_db imagereg.db \
# --new_db database_brief.db \
# --extractor /home/netease/ARWorkspace/maplab_deros/devel/bin/export_desp_to_file \
# --input_model sparse/imagereg \  
# --output_model sparse/brief \
# --summarymap \
# --summap_dir /media/netease/Software/Dataset/Oasis/C6/LocMap/LocMap_0512_night/C6_part_F2N/summary_map \
# --lc_folder /home/netease/ARWorkspace/maplab_deros/src/localizer/share/lcdat \
# --map_builder /home/netease/ARWorkspace/maplab_deros/devel/bin/test_summary_map_creator  \
# --jump external_feature

python3 run_brief_map_build.py \
--colmap_folder /media/netease/Software/Dataset/Oasis/C6/LocMap/C6_part_F2N_FNStairs \
--base_db imagereg.db \
--new_db database_brief.db \
--extractor /home/netease/ARWorkspace/maplab_deros/devel/bin/export_desp_to_file \
--input_model sparse/imagereg \
--output_model sparse/brief \
--summarymap \
--summap_dir /media/netease/Software/Dataset/Oasis/C6/LocMap/C6_part_F2N_FNStairs/summary_map \
--lc_folder /home/netease/ARWorkspace/maplab_deros/src/localizer/share/lcdat \
--map_builder /home/netease/ARWorkspace/maplab_deros/devel/bin/test_summary_map_creator 