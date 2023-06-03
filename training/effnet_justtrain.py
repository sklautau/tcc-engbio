# this is just to train and save the effnet model via tl

#tutorial in part from https://wandb.ai/sayakpaul/efficientnet-tl/reports/Transfer-Learning-With-the-EfficientNet-Family-of-Models--Vmlldzo4OTg1Nw

# this code trains a cnn using an effnet model as a feature extractor

import tensorflow as tf
import tensorflow_hub as hub
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping
import seaborn as sn
import os
import sys
import shutil

plt.close("all")

#%% CHECK ALL VARIABLES AND REPLACE VALUES AS NEEDED
#--------------------------------------------------------------------------
img_size = 300
model_url = 'https://tfhub.dev/google/efficientnet/b3/feature-vector/1'
use_augment = True
type_augment = "Albumentations"
test_id = str(1) #test number
aug_id = str(2) #type of data augmentation
data_id = "1-18k" #ratio
effnet_model = str(3) 
#--------------------------------------------------------------------------

base_name = 'effnetb' + effnet_model + '_augment_teste' + test_id + '_C' + aug_id + '_' + data_id
output_dir = os.path.join('../outputs/albumentations/', base_name + '_output')

if use_augment:
    title_id = 'Augment - ' + type_augment + ' - C' + aug_id + ' - test: ' + test_id + ' -' + ' ratio: ' + data_id
else:
    title_id = 'No augment - ' + 'test: ' + test_id + ' -' + ' ratio: ' + data_id
    

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print("Created folder ", output_dir)
    
#%%
#copy script
copied_script = os.path.join(output_dir, os.path.basename(sys.argv[0]))
shutil.copy2(sys.argv[0], copied_script)
print("Just copied current script as file", copied_script)

def get_training_model(url, trainable=False):
    # Load the respective EfficientNet model but exclude the classification layers
    extractor = hub.KerasLayer(url, input_shape=(img_size, img_size, 3), trainable=trainable)
    
    # Construct the head of the model that will be placed on top of the
    # the base model
    model = tf.keras.models.Sequential([
        extractor,
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(12, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    
    # Compile and return the model
    # model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
    #                       optimizer="adam",
    #                       metrics=["accuracy"])
    
    model.compile(loss='binary_crossentropy', 
                  optimizer="adam",
                  metrics=["accuracy"])
    
    model.summary()
    
    return model

model = get_training_model(model_url) 
print("just got the model")

# Define the folders for train, validation, and test data
train_folder = 'D:/sofia/ufpa/tcc/data_intofolders/albumentations/train-1-C1/'
validation_folder = 'D:/sofia/ufpa/tcc/data_intofolders/val/'
test_folder = 'D:/sofia/ufpa/tcc/data_intofolders/test/'

# Define the image size and other parameters
image_size = (img_size, img_size)
batch_size = 16
epochs = 1000
num_classes = 2

# Loading and preprocessing the training, validation, and test data
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_folder,
    # test_folder,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_folder,
    # test_folder,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_folder,
    # test_folder,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Training the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

# Save the model
model_name = 'model_' + base_name + '.h5'
model.save(os.path.join(output_dir, model_name))
