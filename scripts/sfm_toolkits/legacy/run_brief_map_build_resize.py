
### 建立brief特征对应的地图模型，和summarymap
import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--colmap_folder', required=True)
    parser.add_argument('--scripts_folder', type=str, default='./colmap_data')
    parser.add_argument('--base_db', required=True, help="sfm db with sift")
    parser.add_argument('--new_db', required=True, help="new db with given feature")
    parser.add_argument('--image_prefix', type=str, default='append', help="image subfolder")
    parser.add_argument('--extractor', type=str, default='', help="extractor exe path")
    parser.add_argument('--input_model', required=True)
    parser.add_argument('--output_model', required=True)
    parser.add_argument('--image_ext', type=str, default='jpg')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--jump', type=str, nargs='+', default=[])

    parser.add_argument('--summarymap', action='store_true')
    parser.add_argument('--summap_dir', type=str, default='')
    parser.add_argument('--lc_folder', type=str, default='')
    parser.add_argument('--map_builder', type=str, default='')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    jump = args.jump
    
    dataset_dir = args.colmap_folder
    image_dir = dataset_dir + "/images"

    colmap_scripts_path = args.scripts_folder

    sfm_db = dataset_dir + "/" + args.base_db
    feature_db = dataset_dir + "/" + args.new_db
    extractor = args.extractor

    check_scripts = args.check

    input_model = dataset_dir + "/" + args.input_model

    output_model = dataset_dir + "/" + args.output_model

    lc_folder = args.lc_folder

    map_builder = args.map_builder

    match_file = dataset_dir + "/matches80.txt"

    if not os.path.exists(dataset_dir):
        print("dataset_dir", "not exist.")
        return 
    
    if not os.path.exists(image_dir):
        print("image_dir", "not exist.")
        return 

    if not os.path.isfile(sfm_db):
        print("sfm_db", "not exist.")
        return 
    
    if not os.path.isfile(extractor):
        print("extractor ", extractor, "not exist.")
        return 

    if args.summarymap:
        print("Will output summary map")
        if not os.path.exists(lc_folder):
            print("lc_folder ", lc_folder, "not exist.")
            return 
        if not os.path.isfile(map_builder):
            print("map_builder ", map_builder, "not exist.")
            return 
        summap_dir = args.summap_dir
        if not os.path.isdir(summap_dir):
            os.mkdir(summap_dir)

    if not os.path.exists(input_model):
        print("input_model", "not exist.")
        return 
    
    db = feature_db

    output_model = dataset_dir + "/" + args.output_model
    if not os.path.isdir(output_model):
        os.mkdir(output_model)
    
    tmp_model = input_model + "_reorder"
    if not os.path.isdir(tmp_model):
        os.mkdir(tmp_model)

    print('==================================\nexternal feature extractor\n==================================\n')
    if 'external_feature' not in jump:
        full_command_str = extractor + " " + image_dir + "/" + args.image_prefix \
            + " " + args.image_ext + " 480"
        print(full_command_str)
        os.system(full_command_str)  

        if check_scripts:
            print("finish ", full_command_str, "\npress to continue")
            str = input("")     


    print('==================================\nexternal matcher\n==================================\n')
    if 'external_match' not in jump:
        if os.path.exists(db):
            os.remove(db) 

        full_command_str = 'python3 ' + colmap_scripts_path + "/" + "match_features_with_db_prior.py"  \
            + " --database_file " + sfm_db  \
            + " --feature_ext .jpg.txt" \
            + " --image_dir " + image_dir \
            + " --feature_dir " + image_dir \
            + " --image_prefix " + args.image_prefix \
            + " --match_list_path " + match_file   \
            + " --use_ratio_test "   \
            + " --ratio_test_values 0.8 "
       
        print(full_command_str)
        os.system(full_command_str)  

        if check_scripts:
            print("finish ", full_command_str, "\npress to continue")
            str = input("")    

    print('==================================\ncreate new database\n==================================\n')
    if 'creator' not in jump:
        if os.path.exists(db):
            os.remove(db) 

        full_command_str = 'colmap database_creator --database_path ' + db
        print(full_command_str)
        os.system(full_command_str)  

        if check_scripts:
            print("finish ", full_command_str, "\npress to continue")
            str = input("")    

    print('==================================\nfeature importer\n==================================\n')
    if 'feature' not in jump:
        full_command_str = 'colmap feature_importer --database_path ' + db  \
            + " --ImageReader.single_camera_per_folder 1 " \
            + " --image_path " + image_dir \
            + " --import_path " + image_dir 
        print(full_command_str)
        os.system(full_command_str)  

        if check_scripts:
            print("finish ", full_command_str, "\npress to continue")
            str = input("")      

    print('==================================\nreorder model by database\n==================================\n')
    if 'reorder' not in jump:
        full_command_str = 'python3 ' + colmap_scripts_path + "/" + "colmap_model_modify.py"  \
            + " --database_file " + db  \
            + " --input_model " + input_model \
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
            + " --match_list_path " + match_file  \
            + " --match_type raw"

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

    if args.summarymap and ('summap' not in jump):
        print('==================================\nbuild summary map\n==================================\n')
        full_command_str = 'python3 ' + colmap_scripts_path + "/" + "colmap_read_write_model.py"  \
            + " --input_model " + output_model  \
            + " --input_format .bin" \
            + " --output_model " + output_model \
            + " --output_format .txt" 
        
        print(full_command_str)
        os.system(full_command_str)  
        
        full_command_str = map_builder + " " + lc_folder \
            + " " + image_dir + " " + output_model + " " + summap_dir
        print(full_command_str)
        os.system(full_command_str)  

if __name__ == '__main__':
    main()