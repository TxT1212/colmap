# coding: utf-8
import os
import sys
from fileio.parser_config_info import convert_path_var, convert_path_list
# 外部命令调用接口，暂时只支持python

def run_external_command(name, command, paths):
  command_cvt = convert_path_var(command, command, paths)

  command_type = command_cvt['type']
  program = command_cvt['program']
  prefix = ''

  if command_type == "python":
    prefix = command_cvt['compiler'] + ' '
  else:
    print("Error! Unsupported command type: ", command_type, " of external command: ", name)
    sys.exit()
  
  args = command_cvt['args']
  args_cvt = convert_path_list(args, command_cvt, paths)

  run_command = prefix + program 

  for arg in args_cvt:
    if arg == '':
        arg = '\'\''

    run_command = run_command + ' ' + arg

  print(run_command)

  os.system(run_command)