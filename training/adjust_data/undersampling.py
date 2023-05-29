import pandas as pd
import random 
import numpy as np

csv_train = 'D:/sofia/ufpa/tcc/dataset_updated/train_data.csv'
df_train = pd.read_csv(csv_train)

# csv_test = 'D:/sofia/ufpa/tcc/test_data.csv'
# df_test = pd.read_csv(csv_test)

# csv_val = 'D:/sofia/ufpa/tcc/dataset_updated/val_data.csv'
# df_val = pd.read_csv(csv_val)

# Separate the data by class
class_0_rows = df_train[df_train['target'] == 0]
class_1_rows = df_train[df_train['target'] == 1]

# Determine the desired ratio between the classes
desired_ratio = 2.2

# Calculate the number of instances to keep from the larger class
num_instances_to_keep = len(class_1_rows) * desired_ratio
num_instances_to_keep = np.ceil(num_instances_to_keep)
num_instances_to_keep = int(num_instances_to_keep)

# Randomly select a subset of instances from the larger class
undersampled_class_0_rows = class_0_rows.sample(n=num_instances_to_keep, random_state=42)

# Combine the undersampled class 0 rows with the original class 1 rows
balanced_df = pd.concat([undersampled_class_0_rows, class_1_rows])

# Save the balanced data to a new CSV file
balanced_df.to_csv('train_ratio_2.csv', index=False)
