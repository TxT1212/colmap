
### 用于新增一些图来更新骨架模型，或注册定位地图
import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--colmap_folder', required=True)
    parser.add_argument('--base_db', required=True)
    parser.add_argument('--new_db', required=True)
    parser.add_argument('--input_model', required=True)
    parser.add_argument('--output_model', required=True)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--image_type', type=str, default='individual') #seq or individual
    parser.add_argument('--voc_file', type=str, default='')
    parser.add_argument('--jump', type=str, nargs='+', default=[])

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    jump = args.jump
    
    dataset_dir = args.colmap_folder
    image_dir = dataset_dir + "/images"

    sfm_db = dataset_dir + "/" + args.base_db
    imagereg_db = dataset_dir + "/" + args.new_db

    voc_file = args.voc_file

    check_scripts = args.check

    input_model = dataset_dir + "/" + args.input_model

    output_model = dataset_dir + "/" + args.output_model

    if not os.path.exists(dataset_dir):
        print(dataset_dir, "not exist.")
        return 
    
    if not os.path.exists(image_dir):
        print(image_dir, "not exist.")
        return 

    if not os.path.isfile(sfm_db):
        print(sfm_db, "not exist.")
        return 

    if args.image_type == 'video':
        if not os.path.isfile(voc_file):
            print(voc_file, "not exist.")
            return 

    if not os.path.exists(input_model):
        print(input_model, "not exist.")
        return 
    
    db = imagereg_db

    output_model = dataset_dir + "/" + args.output_model
    if not os.path.isdir(output_model):
        os.mkdir(output_model)

    print('==================================\ncreate new database\n==================================\n')
    if 'creator' not in jump:
        if os.path.exists(db):
            os.remove(db) 

        full_command_str = 'cp ' + sfm_db + " " + db
        print(full_command_str)
        os.system(full_command_str)  

        if check_scripts:
            print("finish ", full_command_str, "\npress to continue")
            str = input("")     

    print('==================================\nfeature extractor\n==================================\n')
    if 'feature' not in jump:
        full_command_str = 'colmap feature_extractor --database_path ' + db  \
            + " --ImageReader.single_camera_per_folder 1 " \
            + " --image_path " + image_dir
        print(full_command_str)
        os.system(full_command_str)  

        if check_scripts:
            print("finish ", full_command_str, "\npress to continue")
            str = input("")            

    if args.image_type == 'video':
        print('==================================\nsequence matches\n==================================\n')
        if 'match' not in jump:
            full_command_str = 'colmap sequential_matcher'  \
                + " --database_path " + db  
            print(full_command_str)
            os.system(full_command_str)  

            full_command_str = 'colmap vocab_tree_matcher'  \
                + " --database_path " + db  \
                + " --VocabTreeMatching.vocab_tree_path " + voc_file \
                + " --SiftMatching.min_num_inliers 50"
            print(full_command_str)
            os.system(full_command_str)  

            if check_scripts:
                print("finish ", full_command_str, "\npress to continue")
                str = input("")       
    else:
        print('==================================\nexhaustive matches\n==================================\n')
        if 'match' not in jump:
            full_command_str = 'colmap exhaustive_matcher'  \
                + " --database_path " + db  
            print(full_command_str)
            os.system(full_command_str)     

    print('==================================\nmapper and BA\n==================================\n')
    if 'mapper' not in jump:
        full_command_str = 'colmap mapper'  \
            + " --database_path " + db  \
            + " --image_path " + image_dir \
            + " --input_path " + input_model \
            + " --output_path " + output_model    

        print(full_command_str)
        os.system(full_command_str)  
        if check_scripts:
            print("finish ", full_command_str, "\npress to continue")
            str = input("")

    print('==================================\nmodel_aligner\n==================================\n')
    if 'aligner' not in jump:

        cp_command_str = 'cp ' + input_model + "/geos.txt" + " " + output_model + "/geos.txt"
        print(cp_command_str)
        os.system(cp_command_str)

        full_command_str = 'colmap model_aligner'  \
            + " --input_path " + output_model \
            + " --output_path " + output_model  \
            + " --ref_images_path " + input_model + "/geos.txt" \
            + " --robust_alignment_max_error 0.05"
        print(full_command_str)
        os.system(full_command_str)  
        if check_scripts:
            print("finish ", full_command_str, "\npress to continue")
            str = input("")
        
        
if __name__ == '__main__':
    main()