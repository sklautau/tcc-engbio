import albumentations as A
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from PIL import Image
import cv2
# from tensorflow.keras.utils import load_img

# Load your unbalanced dataset
# Assume `X` is the feature data and `y` is the corresponding labels

train_folder = 'D:/sofia/ufpa/tcc/data_intofolders/train/1/'
output_folder = 'D:/sofia/ufpa/tcc/data_intofolders/albumentations/train-1-C2-testaug2'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print("Created folder ", output_folder)

desired_width = 300
desired_height = 300

images = []
images_names = []
img_counter = 0
cv_img_clahe = []

for filename in os.listdir(train_folder):
    # images_names.append(os.path.basename(filename))
    images_names.append(os.path.splitext(os.path.basename(filename))[0]) #remove extension
    img_path = os.path.join(train_folder, filename)
    img = image.load_img(img_path, target_size=(desired_width, desired_height))
    #print(img.__class__)
    #print("aqui", list(img.getdata())[0:30])
    img_array = image.img_to_array(img).astype('uint8')
    #print(img_array.__class__)    
    #print("ali", img_array)
    #print(img_array.dtype)
    images.append(img_array)
    print("Read image", images_names[img_counter])
    img_counter += 1
    
    #cv_img = cv2.imread(img_path)
    #cv_img_array = cv_img.astype('uint8')
    #cv_img_clahe.append(cv_img_array)
    
print("# original images = ", img_counter)
#print('was able to read', cv_img)

# Count the number of positive examples
num_positive_samples = len(images)
#%%

# # Assuming you have a list of images as float32 arrays
# # image_list_float32 = [image1_float32, image2_float32, image3_float32, ...]

# # Define a custom transformation using Albumentations Lambda
# def apply_clahe(image):
#     image_uint8 = (image * 255).astype(np.uint8)  # Convert image to uint8
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     image_clahe = clahe.apply(image_uint8)
#     image_float32 = image_clahe.astype(np.float32) / 255  # Convert image back to float32
#     return image_float32

# # # Create the augmentation pipeline
# # transform = A.Compose([
# #     A.Lambda(image=apply_clahe),
# #     # Add other transformations to the pipeline as needed
# # ])

# # Apply the augmentation pipeline to each image
# # transformed_images = [transform(image=image)['image'] for image in img_array]

#%%
# Define the desired total number of (new) positive samples
desired_num_samples = 400

# Calculate the augmentation factor to balance the dataset
augmentation_factor = desired_num_samples // num_positive_samples
print("augmentation_factor =", augmentation_factor)

# Create an augmentation pipeline
augmentation_pipeline = A.Compose([
     # Define your desired augmentations here
     # For example:
     A.CLAHE(),
     # A.Lambda(image=apply_clahe),
     A.HueSaturationValue(hue_shift_limit=[-28,22], #default or -9,20
                          sat_shift_limit=[-30,30],
                          val_shift_limit=[-20,20]), #default
     A.Rotate()
])

# Initialize lists to store augmented data and labels
augmented_X = []
augmented_X_names = []

# Apply augmentation to positive examples
for i in range(len(images)): # was before images
    for j in range(augmentation_factor):
             augmented = augmentation_pipeline(image=images[i])
             augmented_X.append(augmented['image'])
             augmented_X_names.append(images_names[i] + '_' + str(j) + '.jpg')

print("Have now", len(augmented_X))

#image.save('output.jpg', 'JPEG')  # Replace 'output.jpg' with the desired output file name

# Convert augmented data and labels to numpy arrays

augmented_X = np.array(augmented_X)
print(augmented_X[0].__class__)
print(augmented_X[0].shape)
print(np.max(augmented_X[0]))
print(np.min(augmented_X[0]))


for i in range(len(augmented_X)):  
    PIL_image = Image.fromarray(np.uint8(augmented_X[i]))
    PIL_image.save((os.path.join(output_folder, augmented_X_names[i])), 'JPEG')
    print("Wrote file", augmented_X_names[i])

# PIL_image = Image.fromarray(numpy_image.astype('uint8'), 'RGB')


# Combine original data with augmented data
#balanced_X = np.concatenate((X, augmented_X), axis=0)
#balanced_y = np.concatenate((y, augmented_y), axis=0)
