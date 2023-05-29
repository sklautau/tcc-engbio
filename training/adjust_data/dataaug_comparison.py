#for comparison with data aug images - albumentations
# this code resizes the original images and saves them

from tensorflow.keras.preprocessing import image
import os
from PIL import Image
import numpy as np

train_folder = 'D:/sofia/ufpa/tcc/data_intofolders/train/0/'
output_folder = 'D:/sofia/ufpa/tcc/data_intofolders/albumentations/train-1-C1/0/'

desired_width = 224
desired_height = 224

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print("Created folder ", output_folder)

# print("# original images = ", img_counter)

# Iterate over files in the input directory
for filename in os.listdir(train_folder):
    input_path = os.path.join(train_folder, filename)
    output_path = os.path.join(output_folder, filename)
    
    
    # Load and resize the image
    img = image.load_img(input_path, target_size=(desired_width, desired_width))

    # Save the resized image
    img.save(output_path)
    print("Writing to file: ", filename)