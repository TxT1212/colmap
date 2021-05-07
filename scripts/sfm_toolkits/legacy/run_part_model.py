
### 用于创建从一个骨架大模型中分拆出的局部模型，并使得子模型的坐标系与大模型统一
import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--colmap_folder', required=True)
    parser.add_argument('--input_model', required=True, help="origin whole model")
    parser.add_argument('--match_file', required=True)
    parser.add_argument('--scripts_folder', type=str, default='./colmap_data')
    parser.add_argument('--output_model', required=True)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--jump', type=str, nargs='+', default=[])

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    jump = args.jump

    dataset_dir = args.colmap_folder
    image_dir = dataset_dir + "/images"

    colmap_scripts_path = args.scripts_folder

    check_scripts = args.check

    gt_full_model= dataset_dir + "/" + args.input_model

    match_file= dataset_dir + "/" + args.match_file

    db = dataset_dir + "/database.db"

    if not os.path.exists(dataset_dir):
        print(dataset_dir, "not exist.")
        return 
    
    if not os.path.exists(image_dir):
        print(image_dir, "not exist.")
        return 

    if not os.path.exists(colmap_scripts_path):
        print(colmap_scripts_path, "not exist.")
        return 

    if not os.path.isfile(match_file):
        print(match_file, "not exist.")
        return 
    
    if not os.path.exists(gt_full_model):
        print(gt_full_model, "not exist.")
        return 
    
    tmp_model = gt_full_model + "_reorder"
    if not os.path.isdir(tmp_model):
        os.mkdir(tmp_model)
    
    output_model = dataset_dir + "/" + args.output_model
    if not os.path.isdir(output_model):
        os.mkdir(output_model)
    
    print('==================================\ncreate new database\n==================================\n')
    if 'creator' not in jump:
        # db = dataset_dir + "/database.db"
        full_command_str = 'colmap database_creator --database_path ' + db
        print(full_command_str)
        os.system(full_command_str)  

        if check_scripts:
            print("finish ", full_command_str, "\npress to continue")
            str = input("")       

    print('==================================\nfeature extractor\n==================================\n')
    if 'feature' not in jump:
        
        full_command_str = 'colmap feature_extractor --database_path ' + db  \
            + " --ImageReader.single_camera  1 " \
            + " --image_path " + image_dir
        print(full_command_str)
        os.system(full_command_str)  

        if check_scripts:
            print("finish ", full_command_str, "\npress to continue")
            str = input("")       

    print('==================================\nreorder model by database\n==================================\n')
    if 'reorder' not in jump:
        full_command_str = 'python3 ' + colmap_scripts_path + "/" + "colmap_model_modify.py"  \
            + " --database_file " + db  \
            + " --input_model " + gt_full_model \
            + " --output_model " + tmp_model

        print(full_command_str)
        os.system(full_command_str)  
        if check_scripts:
            print("finish ", full_command_str, "\npress to continue")
            str = input("")       

    print('==================================\nimport matches\n==================================\n')
    if 'match' not in jump:
        full_command_str = 'colmap matches_importer'  \
            + " --database_path " + db  \
            + " --match_list_path " + match_file 

        print(full_command_str)
        os.system(full_command_str)  
        if check_scripts:
            print("finish ", full_command_str, "\npress to continue")
            str = input("")       

    print('==================================\ntriangulator\n==================================\n')
    if 'triangulator' not in jump:
        full_command_str = 'colmap point_triangulator'  \
            + " --database_path " + db  \
            + " --image_path " + image_dir \
            + " --input_path " + tmp_model \
            + " --output_path " + output_model    

        print(full_command_str)
        os.system(full_command_str)  
        if check_scripts:
            print("finish ", full_command_str, "\npress to continue")
            str = input("")       

    print('==================================\nmapper and BA\n==================================\n')
    if 'mapper' not in jump:
        full_command_str = 'colmap mapper'  \
            + " --database_path " + db  \
            + " --image_path " + image_dir \
            + " --input_path " + output_model \
            + " --output_path " + output_model    

        print(full_command_str)
        os.system(full_command_str)  
        if check_scripts:
            print("finish ", full_command_str, "\npress to continue")
            str = input("")

    print('==================================\nmodel_aligner\n==================================\n')
    if 'aligner' not in jump:
        full_command_str = 'colmap model_aligner'  \
            + " --input_path " + output_model \
            + " --output_path " + output_model  \
            + " --ref_images_path " + gt_full_model + "/geos.txt" \
            + " --robust_alignment_max_error 0.05"
        print(full_command_str)
        os.system(full_command_str)  
        if check_scripts:
            print("finish ", full_command_str, "\npress to continue")
            str = input("")

        cp_command_str = 'cp ' + gt_full_model + "/geos.txt" + " " + output_model + "/geos.txt"
        print(cp_command_str)
        os.system(cp_command_str)

if __name__ == '__main__':
    main()