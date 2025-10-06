# -*- coding: utf-8 -*-
import os
import shutil


def remove_hidden_items(folder):
    for root, dirs, files in os.walk(folder, topdown=False):
        for file in files:
            if file.startswith('._') or file == '.DS_Store':
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                        print(f'Removed hidden file: {file_path}')
                    except Exception as e:
                        print(f'Error removing file {file_path}: {e}')

        for dir in dirs:
            if dir.startswith('._') or dir == '__pycache__':
                dir_path = os.path.join(root, dir)
                if os.path.isdir(dir_path):
                    try:
                        shutil.rmtree(dir_path)
                        print(f'Removed directory: {dir_path}')
                    except Exception as e:
                        print(f'Error removing directory {dir_path}: {e}')


# if __name__ != '__main__':
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
remove_hidden_items(base_dir)