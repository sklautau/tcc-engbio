import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import os

# Load the CSV file
# attention: all the JPG files must be in the same directory as the CSV file
csv_file = 'D:/sofia/ufpa/tcc/1000_jpg/train_jpg/traincsv.csv'
# csv_file = 'D:\\sofia\\ufpa\\tcc\\1000_jp\\train_jpg\\traincsv.csv'
df = pd.read_csv(csv_file, dtype=str)

df['image_name'] = df['image_name']+'.jpg'

directory='D:/sofia/ufpa/tcc/dataset_updated/val/'


# Create a list to store the names of the JPG files in the folder
jpg_files = []

# Iterate through the files in the folder
for filename in os.listdir(directory):
    # Check if the file extension is ".jpg"
    if filename.lower().endswith(".jpg"):
        # Add the file name to the list of JPG files
        jpg_files.append(filename)
        

i = 0
# Loop over the rows in the dataframe
for index, row in df.iterrows():
    # Get the filename from the "filename" column of the row
    filename = row["image_name"]
    if filename in jpg_files:
        # Convert the row to a string with columns separated by comma
        row_string = ','.join(str(x) for x in row)
        print(row_string)
        #print(row.to_string(index=False, header=False),sep=', ')\
        i += 1

print("Found", i, "files")
            