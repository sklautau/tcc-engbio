import os
import shutil
import pandas as pd


#df_train = pd.read_csv(csv_train, dtype=str)

# # csv_test = 'D:/sofia/ufpa/tcc/dataset_updated/test_data.csv'
# df_test = pd.read_csv(csv_test, dtype=str)

# # csv_val = 'D:/sofia/ufpa/tcc/dataset_updated/val_data.csv'
# df_val = pd.read_csv(csv_val, dtype=str)

# Load CSV file into a pandas DataFrame
#df = pd.read_csv('file_list.csv')

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

train_folder = 'D:/sofia/ufpa/tcc/data_intofolders/val/'
source_folder = 'D:/sofia/ufpa/tcc/dataset_updated/val/'
csv_train = 'D:/sofia/ufpa/tcc/dataset_updated/val_data.csv'

copytofolder(csv_train,train_folder,source_folder)