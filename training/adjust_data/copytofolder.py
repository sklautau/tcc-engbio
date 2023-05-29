import os
import shutil
import pandas as pd

def copytofolder(filename,folder_name,source_folder):
    df = pd.read_csv(filename, dtype=str)

    # Create directory for each class
    for class_name in df['target'].unique():
        fullpath = folder_name + class_name    
        os.makedirs(fullpath, exist_ok=True)
        # print(fullpath)
        
    # Copy files to respective directories
    for index, row in df.iterrows():
        src_file = row['image_name']
        class_name = row['target']
        fullpath_src = source_folder + src_file
        fullpath = folder_name + class_name    
        # print(fullpath_src)
        # print(fullpath)
        shutil.copy(fullpath_src, fullpath)

train_folder = 'D:/sofia/ufpa/tcc/data_intofolders_balanced/train_ratio2/'
source_folder = 'D:/sofia/ufpa/tcc/dataset_updated/train/'
csv_train = 'D:/sofia/ufpa/tcc/code/train_ratio_2.csv'

copytofolder(csv_train,train_folder,source_folder)