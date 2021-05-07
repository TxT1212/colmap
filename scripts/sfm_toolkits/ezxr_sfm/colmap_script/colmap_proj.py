# coding: utf-8
import types
import os


class ColmapProj:
    # colmap项目的文件夹路径，图片路径，相机模型是共用的
    # 具体环节中所用的database/feature/model是不一样的
    model_proj_path = ''
    model_image_path = ''
    colmap_app_path = ''
    cameras = []
    path_inited = False
    global_paths = []
    
    @staticmethod
    def init_colmap_env(global_path, camera_struct):
        ColmapProj.model_proj_path = global_path['model_proj_path']
        ColmapProj.model_image_path = global_path['model_proj_image_path']
        ColmapProj.cameras = camera_struct
        ColmapProj.colmap_app_path = global_path['colmap_app_path']
        ColmapProj.global_paths = global_path
        ColmapProj.path_inited = True
    
    @staticmethod
    def colmap_param_parser(param_name, param_value):
        param_value_str = ""
        if type(param_value) == type('a'):
            param_value_str = param_value 
        elif type(param_value) == type(True):
            param_value_str = str(int(param_value)) 
        else:
            param_value_str = str(param_value) 
        return param_value_str

    @staticmethod
    def parse_colmap_command(command_name, command_params, colmap_app_path=None):
        param_str_list = []
        for param in command_params:
            param_value_str = ColmapProj.colmap_param_parser(param, command_params[param])
            param_str = "--" + param + " " + param_value_str
            param_str_list.append(param_str)

        splint = ' '

        if not (colmap_app_path==None):
            colmap_command_str = colmap_app_path + " " + command_name + " " + splint.join(param_str_list)
        else:
            colmap_command_str = ColmapProj.colmap_app_path + " " + command_name + " " + splint.join(param_str_list)
       # print(colmap_command_str)
        return colmap_command_str


    def __init__(self, colmap_json_struct):
        if not ColmapProj.path_inited:
            print("Error, colmap paths haven't been set.")
            exit 

        if 'database' in colmap_json_struct.keys():
            self.model_database_path = colmap_json_struct['database']
        else:
            print("Missing database.\n")
        
        if 'input_model' in colmap_json_struct.keys():
            self.input_model = colmap_json_struct['input_model']
        else:
            print("Missing input_model.\n")

        if 'output_model' in colmap_json_struct.keys():
            self.output_model = colmap_json_struct['output_model']
        else:
            print("Missing output_model.\n")

        if not os.path.isdir(self.output_model):
            os.makedirs(self.output_model)
    
    # def colmap_param_parser(self, param_name, param_value):
    #     param_value_str = ""
    #     if type(param_value) == type('a'):
    #         param_value_str = param_value 
    #     elif type(param_value) == type(True):
    #         param_value_str = str(int(param_value)) 
    #     else:
    #         param_value_str = str(param_value) 
    #     return param_value_str

    # def parse_colmap_command(self, command_name, command_params):
    #     param_str_list = []
    #     for param in command_params:
    #         param_value_str = self.colmap_param_parser(param, command_params[param])
    #         param_str = "--" + param + " " + param_value_str
    #         param_str_list.append(param_str)

    #     splint = ' '
    #     colmap_command_str = ColmapProj.colmap_app_path + " " + command_name + " " + splint.join(param_str_list)
    #    # print(colmap_command_str)
    #     return colmap_command_str

