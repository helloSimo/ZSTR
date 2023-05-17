import glob
import os
import jsonlines

for dir_name in glob.glob('model_{}_{}*{}*'.format('dpr', '', '1e5')):
    print(dir_name)
