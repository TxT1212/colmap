
dataset_dir=$1
sfm_db=${dataset_dir}/database_sfm_geo.db
imagereg_db=${dataset_dir}/imagereg.db
image_dir=${dataset_dir}/images
gt_model=${dataset_dir}/sparse/geo
output_model=${dataset_dir}/sparse/imagereg
vocab_path=/home/administrator/Oasis/colmap

#100-1000images
#vocab = "${vocab_path}/vocab_tree_flickr100K_words32K.bin"

#1000-10000images
vocab="${vocab_path}/vocab_tree_flickr100K_words256K.bin"

#10000+images
#vocab = "${vocab_path}/vocab_tree_flickr100K_words1M.bin"

# 先将一部分图像文件拷贝到images文件夹下
# 拷贝原始的db文件
rm ${imagereg_db}

cp ${sfm_db} ${imagereg_db}

# 特征提取和特征匹配
colmap feature_extractor \
--database_path ${imagereg_db} \
--image_path ${image_dir}

colmap exhaustive_matcher  --database_path ${imagereg_db}  

# colmap vocab_tree_matcher \
# --database_path ${imagereg_db}  \
# --VocabTreeMatching.vocab_tree_path ${vocab}

# colmap sequential_matcher \
# --database_path ${imagereg_db}  \
# --SequentialMatching.vocab_tree_path ${vocab} \
# --SequentialMatching.loop_detection 1 \
# --SequentialMatching.loop_detection_num_images 100 \
# --SequentialMatching.loop_detection_num_nearest_neighbors 5 


# colmap transitive_matcher \
# --database_path ${imagereg_db}  \

# 图像配准和三角化
mkdir ${output_model}
colmap image_registrator \
--database_path ${imagereg_db} \
--input_path ${gt_model} \
--output_path ${output_model} \
--Mapper.min_num_matches 200 \
--Mapper.ignore_watermarks 1 \
--Mapper.tri_min_angle 10

python3 colmap_read_write_model.py  \
--input_model ${output_model} \
--input_format .bin \
--output_model ${output_model} \
--output_format .txt \
--poseonly

exit 1
