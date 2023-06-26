'''
To reduce computational cost, save the output of the backend NN.
Find models at https://tfhub.dev/google/collections/efficientnet_v2/1
'''

from math import exp
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
#import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
#import sklearn.metrics 
#from sklearn.metrics import confusion_matrix, roc_curve, auc, recall_score, f1_score, precision_score, precision_recall_curve
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import seaborn as sn
import os
import sys
import shutil
import pickle
#import argparse
import pandas as pd
from tensorflow.keras.applications.resnet import ResNet152, preprocess_input
from tensorflow.keras.callbacks import TensorBoard
import urllib
import warnings
from tensorflow.keras import regularizers

import optuna
from optuna.integration import TFKerasPruningCallback

from keras.backend import clear_session
from keras.datasets import mnist
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

#To avoid the warning in
#https://github.com/tensorflow/tensorflow/issues/47554
from absl import logging
logging.set_verbosity(logging.ERROR)

#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)
if False:
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    def fix_gpu():
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)

    fix_gpu()

# Global variables
JPG_SPLIT_INTO_0_1_SUBDIRECTORIES = True
if JPG_SPLIT_INTO_0_1_SUBDIRECTORIES:
    #folders with augmentations
    #TRAIN_FOLDER = 'D:/sofia/ufpa/tcc/data_intofolders/albumentations/train-1-C1'
    TRAIN_FOLDER = 'D:/sofia/ufpa/tcc/data_intofolders/albumentations/train-1-C3'
else:
    TRAIN_FOLDER = 'D:/sofia/ufpa/tcc/dataset_updated/train'

# Choose if balanced or not
NUM_TRAIN_EXAMPLES = 3000
if False:  
    # we do not use augmentation, so the number of positive examples is restricted to 324
    NUM_DESIRED_NEGATIVE_TRAINING_EXAMPLES = NUM_TRAIN_EXAMPLES-324
else:
    # we use augmentation and created augmented positive examples. We can use a balanced set then
    #NUM_DESIRED_NEGATIVE_TRAINING_EXAMPLES = int(NUM_TRAIN_EXAMPLES/2)
    NUM_DESIRED_NEGATIVE_TRAINING_EXAMPLES = int(3*NUM_TRAIN_EXAMPLES/4)

# Choose the model
if True:
    #Model with 7 M parameters
    MODEL_URL = 'https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b1/feature_vector/2'
    MODEL_NAME = 'efficientnet_v2_imagenet1k_b1'
    NUM_PIXELS = 240
else:
    #Model with 200 M parameters
    MODEL_URL = 'https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_xl/feature_vector/2'
    MODEL_NAME = 'efficientnet_v2_imagenet21k_ft1k_xl'
    NUM_PIXELS = 512 # Define the input shape of the images

#Important: output folder
ID = str(1)
if JPG_SPLIT_INTO_0_1_SUBDIRECTORIES:
    #last_folder = os.path.basename(os.path.dirname(TRAIN_FOLDER))
    last_folder = os.path.basename(TRAIN_FOLDER)
    OUTPUT_DIR = '../backend_output/aug_0_1_' + last_folder + '/' + MODEL_NAME + '_N' + str(NUM_TRAIN_EXAMPLES) + '_id_' + ID + '/' #os.path.join('../outputs/unbalanced/id_' + str(simulation_ID), base_name)        
else:
    OUTPUT_DIR = '../backend_output/' +  MODEL_NAME + '_N' + str(NUM_TRAIN_EXAMPLES) + '_id_' + ID + '/' #os.path.join('../outputs/unbalanced/id_' + str(simulation_ID), base_name)        
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print("Created folder ", OUTPUT_DIR)
VERBOSITY_LEVEL = 0 #use 1 to see the progress bar when training and testing
IMAGESIZE = (NUM_PIXELS, NUM_PIXELS)      # Define the input shape of the images
INPUTSHAPE = (NUM_PIXELS, NUM_PIXELS, 3)  # NN input

def save_outputs(model, generator, dataset_type):
    y_true = np.array(generator.classes)
    X = model.predict(generator, verbose=VERBOSITY_LEVEL)

    examples = (X, y_true)

    print("Shapes:")
    print(examples[0].shape, examples[1].shape)

    # https://stackoverflow.com/questions/41061457/keras-how-to-save-the-training-history-attribute-of-the-history-object
    pickle_file_path = os.path.join(OUTPUT_DIR, dataset_type + '.pickle')
    with open(pickle_file_path, 'wb') as file_pi:
        pickle.dump(examples, file_pi)
    print("Wrote", pickle_file_path)

def get_all_jpg_files_under_folder(root_folder):
    all_image_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            file = str(file).lower()
            if file.endswith(".jpg"):
                all_image_files.append(os.path.join(root, file))
    return all_image_files

def get_data_frame_from_directory(root_folder):
    #for negative examples
    neg_path = os.path.join(root_folder, '0')
    if not os.path.isdir(neg_path):
        raise Exception('Folder ' + neg_path + ' does not exist!')
    neg_files = get_all_jpg_files_under_folder(neg_path)
    pos_path = os.path.join(root_folder, '1')
    if not os.path.isdir(pos_path):
        raise Exception('Folder ' + pos_path + ' does not exist!')
    pos_files = get_all_jpg_files_under_folder(pos_path)

    # create labels, concatenating two lists, one with 0`s and another with 1`s
    y = ['0']*len(neg_files) + ['1']*len(pos_files)

    all_files = list(neg_files) + list(pos_files)

    # Create the DataFrame with same header as csv files
    #df = pd.DataFrame({'image_name': all_files}) #, 'target': y})

  
    # Calling DataFrame constructor after zipping
    # both lists, with columns specified
    df = pd.DataFrame(list(zip(all_files, y)),
                columns =['image_name', 'target'])

    return df

def get_data_generators_from_dataframe(num_desired_negative_train_examples, batch_size):
    # Define the folders for train, validation, and test data
    #PC:
    validation_folder = 'D:/sofia/ufpa/tcc/dataset_updated/val'
    test_folder = 'D:/sofia/ufpa/tcc/dataset_updated/test'

    #define the dataframe for training files
    if JPG_SPLIT_INTO_0_1_SUBDIRECTORIES:
        traindf = get_data_frame_from_directory(TRAIN_FOLDER)
        #TO DO make sure the correct positives are included?
        traindf = decrease_num_negatives_and_positives(traindf, num_desired_negative_train_examples)
        train_folder = None #use absolute path
    else:
        train_folder = TRAIN_FOLDER
        train_csv = 'D:/sofia/ufpa/tcc/train_data.csv'
        #do not remove header
        traindf=pd.read_csv(train_csv,dtype=str)
        traindf = decrease_num_negatives(traindf, num_desired_negative_train_examples)

    test_csv = 'D:/sofia/ufpa/tcc/test_data.csv'
    validation_csv = 'D:/sofia/ufpa/tcc/val_data.csv'

    #do not remove header
    testdf=pd.read_csv(test_csv,dtype=str)
    validationdf=pd.read_csv(validation_csv,dtype=str)

    testdf = decrease_num_negatives(testdf, 184)
    validationdf = decrease_num_negatives(validationdf, 76)

    if False:
        print('train:')
        print(traindf['target'].value_counts())
        print('test:')
        print(testdf['target'].value_counts())
        print('validation:')
        print(validationdf['target'].value_counts())

    train_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
            dataframe = traindf,
            directory = train_folder,
            x_col="image_name", 
            y_col="target",
            target_size=IMAGESIZE,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )        

    # Loading and preprocessing the training, validation, and test data
    validation_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    validation_generator = validation_datagen.flow_from_dataframe(
            dataframe = validationdf,
            directory = validation_folder,
            x_col="image_name", 
            y_col="target",
            target_size=IMAGESIZE,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
    )

    test_generator = test_datagen.flow_from_dataframe(
            dataframe = testdf,
            directory = test_folder,
            x_col="image_name", 
            y_col="target",
            target_size=IMAGESIZE,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
    )
    return train_generator, validation_generator, test_generator

# not working! CURRENT_MODEL is None
def save_best_model_callback(study, trial):
    global BEST_MODEL, OUTPUT_DIR
    best_model_name = "optuna_best_model" # do not use .h5 extension to save in modern format
    best_model_name = os.path.join(OUTPUT_DIR, best_model_name)
    if study.best_trial == trial:
        #BEST_MODEL = CURRENT_MODEL
        print("Saving best model ", best_model_name, "...")
        #BEST_MODEL.save(best_model_name)
    

def save_backend_outputs():
    batch_size = 1
    train_generator, validation_generator, test_generator = get_data_generators_from_dataframe(NUM_DESIRED_NEGATIVE_TRAINING_EXAMPLES, batch_size)

    # Define the CNN model
    model = Sequential()

    #Find models at https://tfhub.dev/google/collections/efficientnet_v2/1
    # Load the respective EfficientNet model but exclude the classification layers
    trainable = False
    model_url = MODEL_URL
    extractor = hub.KerasLayer(model_url, input_shape=INPUTSHAPE, trainable=trainable)

    model.add(extractor)
    
    model.summary()
    save_outputs(model, train_generator, "train")
    save_outputs(model, test_generator, "test")
    save_outputs(model, validation_generator, "validation")

def decrease_num_negatives_and_positives(df, desired_num_negative_examples):
    '''
    Create dataframe with desired_num_rows rows from df
    '''
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    neg_examples = shuffled_df[shuffled_df['target'] == '0'].copy()
    neg_examples = neg_examples.head( round(desired_num_negative_examples) ).copy()

    num_positive_examples = NUM_TRAIN_EXAMPLES - desired_num_negative_examples
    pos_examples = shuffled_df[shuffled_df['target'] == '1'].copy()    
    pos_examples = pos_examples.head( num_positive_examples ).copy()

    newdf = pd.concat([neg_examples, pos_examples], ignore_index=True)
    newdf = newdf.sample(frac=1).reset_index(drop=True) #shuffle again
    return newdf

def decrease_num_negatives(df, desired_num_negative_examples):
    '''
    Create dataframe with desired_num_rows rows from df
    '''
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    neg_examples = shuffled_df[shuffled_df['target'] == '0'].copy()
    neg_examples = neg_examples.head( round(desired_num_negative_examples) ).copy()

    pos_examples = shuffled_df[shuffled_df['target'] == '1'].copy()    
    newdf = pd.concat([neg_examples, pos_examples], ignore_index=True)
    newdf = newdf.sample(frac=1).reset_index(drop=True) #shuffle again
    return newdf


if __name__ == '__main__':
    print("=====================================")
    print("Save backend outputs to files")
    #copy script
    copied_script = os.path.join(OUTPUT_DIR, os.path.basename(sys.argv[0]))
    shutil.copy2(sys.argv[0], copied_script)
    print("Just copied current script as file", copied_script)

    save_backend_outputs()
