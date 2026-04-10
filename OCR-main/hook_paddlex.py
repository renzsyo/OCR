# hook-paddlex.py
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect ALL yaml/json/txt config files PaddleX needs
datas = collect_data_files('paddlex', includes=['**/*.yaml', '**/*.yml',
                                                 '**/*.json', '**/*.txt',
                                                 '**/*.pbtxt'])
hiddenimports = collect_submodules('paddlex')