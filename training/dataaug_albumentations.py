import albumentations as A
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from PIL import Image
# from tensorflow.keras.utils import load_img


# Load your unbalanced dataset
# Assume `X` is the feature data and `y` is the corresponding labels

train_folder = 'D:/sofia/ufpa/tcc/data_intofolders/train/1'
output_folder = 'D:/sofia/ufpa/tcc/data_intofolders/albumentations/train-1-C1'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print("Created folder ", output_folder)

desired_width = 224
desired_height = 224

images = []
images_names = []
img_counter = 0

for filename in os.listdir(train_folder):
    # images_names.append(os.path.basename(filename))
    images_names.append(os.path.splitext(os.path.basename(filename))[0]) #remove extension
    img_path = os.path.join(train_folder, filename)
    img = image.load_img(img_path, target_size=(desired_width, desired_height))
    print(img.__class__)
    img_array = image.img_to_array(img)
    print(img_array.__class__)    
    images.append(img_array)
    print("Read image", images_names[img_counter])
    img_counter += 1

print("# original images = ", img_counter)

# Count the number of positive examples
num_positive_samples = len(images)

# Define the desired total number of (new) positive samples
desired_num_samples = 18227

# Calculate the augmentation factor to balance the dataset
augmentation_factor = desired_num_samples // num_positive_samples
print("augmentation_factor =", augmentation_factor)

# Create an augmentation pipeline
augmentation_pipeline = A.Compose([
     # Define your desired augmentations here
     # For example:
     A.RandomRotate90(),
     A.HorizontalFlip(),
     A.VerticalFlip()
])

# Initialize lists to store augmented data and labels
augmented_X = []
augmented_X_names = []

# Apply augmentation to positive examples
for i in range(len(images)):
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
