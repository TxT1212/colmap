if [ "" = "$word" ] ;then
    dataset_dir=.
else
    dataset_dir=$1
fi
sfm_db=${dataset_dir}/database.db
imagereg_db=${dataset_dir}/imagereg.db
image_dir=${dataset_dir}/images
gt_model=${dataset_dir}/sparse/part_model
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

# # 特征提取和特征匹配
colmap feature_extractor \
--database_path ${imagereg_db} \
--ImageReader.single_camera_per_folder  1 \
--image_path ${image_dir}

colmap sequential_matcher \
--database_path ${imagereg_db} 

colmap vocab_tree_matcher \
--database_path ${imagereg_db}  \
--VocabTreeMatching.vocab_tree_path ${vocab} \
--SiftMatching.min_num_inliers 50


# colmap exhaustive_matcher  --database_path ${imagereg_db}

mkdir ${output_model}
colmap mapper \
    --database_path ${imagereg_db} \
    --image_path ${image_dir} \
    --input_path ${gt_model} \
    --output_path ${output_model}

colmap model_aligner \
    --input_path ${output_model} \
    --output_path ${output_model} \
    --ref_images_path ${gt_model}/geos.txt \
    --robust_alignment_max_error 0.01
