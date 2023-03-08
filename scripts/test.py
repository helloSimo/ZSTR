import re
import os
from glob import glob

file_list = glob('model_dpr*/setting.txt')
print(file_list)
print(len(file_list))
file_list = glob('model_dpr*')
print(file_list)
print(len(file_list))