### 用于创建从一个大模型中分拆出的局部模型，并使得坐标系与大模型统一
if [ "" = "$word" ] ;then
    dataset_dir=.
else
    dataset_dir=$1
fi

image_dir=${dataset_dir}/images
db=${dataset_dir}/database.db
colmap_scripts_path=/home/mm/ARWorkspace/dataset_collection/sfm_data_toolkit/colmap_data
gt_full_model=${dataset_dir}/models/full_aligned_scaled


#match文件可直接从原大模型中复制，colmap会自动跳过不存在的图像
match_file=${dataset_dir}/match_image_list.txt   

colmap database_creator --database_path ${db}

colmap feature_extractor \
--database_path ${db} \
--ImageReader.single_camera  1 \
--image_path ${image_dir}

mkdir ${dataset_dir}/models/
mkdir ${dataset_dir}/models/full_model

python3 ${colmap_scripts_path}/colmap_model_modify.py  \
--database_file ${db} \
--input_model ${gt_full_model} \
--output_model ${dataset_dir}/models/full_model 

colmap matches_importer \
--database_path ${db} \
--match_list_path ${match_file}

mkdir ${dataset_dir}/models/part_model
colmap point_triangulator \
    --database_path ${db} \
    --image_path "${image_dir}" \
    --input_path ${dataset_dir}/models/full_model  \
    --output_path ${dataset_dir}/models/part_model


colmap mapper \
    --database_path ${db} \
    --image_path "${image_dir}" \
    --input_path ${dataset_dir}/models/part_model  \
    --output_path ${dataset_dir}/models/part_model

colmap model_aligner \
    --input_path ${dataset_dir}/models/part_model  \
    --output_path ${dataset_dir}/models/part_model  \
    --ref_images_path ${gt_full_model}/geos.txt \
    --robust_alignment_max_error 0.05
