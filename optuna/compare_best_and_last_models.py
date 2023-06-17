'''
Get performance from last model from files trainHistoryDict.pickle, which store the history, including data from training and test stages
and compare with the saved best model.
'''
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#additional global variables
num_desired_negative_train_examples = 324
IMAGESIZE = (240, 240)      # Define the input shape of the images
VERBOSITY_LEVEL = 0

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

def get_data_generators(num_desired_negative_train_examples, batch_size):
    # Define the folders for train, validation, and test data
    #PC:
    train_folder = 'D:/sofia/ufpa/tcc/dataset_updated/train'
    validation_folder = 'D:/sofia/ufpa/tcc/dataset_updated/val'
    test_folder = 'D:/sofia/ufpa/tcc/dataset_updated/test'

    train_csv = 'D:/sofia/ufpa/tcc/train_data.csv'
    test_csv = 'D:/sofia/ufpa/tcc/test_data.csv'
    validation_csv = 'D:/sofia/ufpa/tcc/val_data.csv'

    #do not remove header
    traindf=pd.read_csv(train_csv,dtype=str)
    testdf=pd.read_csv(test_csv,dtype=str)
    validationdf=pd.read_csv(validation_csv,dtype=str)

    traindf = decrease_num_negatives(traindf, num_desired_negative_train_examples)

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
            shuffle=True
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
            shuffle=True
    )

    test_generator = test_datagen.flow_from_dataframe(
            dataframe = testdf,
            directory = test_folder,
            x_col="image_name", 
            y_col="target",
            target_size=IMAGESIZE,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=True
    )
    return train_generator, validation_generator, test_generator

def find_pkl_files(folder):
    pkl_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith("trainHistoryDict.pickle"):
                pkl_files.append(os.path.join(root, file))
    return pkl_files

def compare_models(root_folder):
    metric = 'val_acc'
    pkl_files = find_pkl_files(root_folder)
    batch_size = 10
    train_generator, validation_generator, test_generator = get_data_generators(num_desired_negative_train_examples, batch_size)

    best_model_val_acc = list()
    last_model_val_acc = list()
    best_model_val_auc = list()
    last_model_val_auc = list()
        
    for file in pkl_files:
        #print(file)
        this_path = os.path.dirname(file)
        with open(file, "rb") as file_pi:
            history = pickle.load(file_pi)
            val_accuracy = history['val_accuracy'][-1]
            val_auc = history['val_auc'][-1]

        last_model_val_acc.append(val_accuracy)
        last_model_val_auc.append(val_auc)

        # Read the best model
        # add custom_objects according to https://stackoverflow.com/questions/61814614/unknown-layer-keraslayer-when-i-try-to-load-model
        model = load_model(this_path, custom_objects={'KerasLayer':hub.KerasLayer})

        validation_generator.reset()
        val_loss2, val_accuracy2, val_auc2 = model.evaluate(validation_generator, verbose=VERBOSITY_LEVEL)

        best_model_val_acc.append(val_accuracy2)
        best_model_val_auc.append(val_auc2)

        #print("val_accuracy, val_auc, val_accuracy2, val_auc2")
        #print(val_accuracy, val_auc, val_accuracy2, val_auc2)
        print("Best - last: val_accuracy2-val_accuracy,  val_auc2-val_auc")
        print(val_accuracy2-val_accuracy,  val_auc2-val_auc)

    plt.close("all")
    output_file_name = os.path.join(root_folder, 'best_last_accuracy.png')
    create_plot(best_model_val_acc, last_model_val_acc, "val_accuracy")
    plt.savefig(output_file_name)
    print("Wrote", output_file_name)

    plt.close("all")
    output_file_name = os.path.join(root_folder, 'best_last_auc.png')
    create_plot(best_model_val_auc, last_model_val_auc, "val_AUC")
    plt.savefig(output_file_name)
    print("Wrote", output_file_name)

def create_plot(best_model_val, last_model_val, metric):
    print(metric, 'best_model_val_acc', best_model_val, sep=",")
    print(metric, 'last_model_val_acc', last_model_val, sep=",")
    N = len(last_model_val)
    plt.plot(np.arange(N)+1, last_model_val, '-x', np.arange(N)+1, best_model_val, '--o')
    plt.xlabel('trial #')
    plt.ylabel(metric)
    plt.legend(["Last","Best"])    
    #plt.show()
    
if __name__ == '__main__':

    print("=====================================")
    print("Collect results")

    parser = argparse.ArgumentParser()
    #required arguments    
    parser.add_argument('--root_folder', help='Parent folder for file(s) with name trainHistoryDict.pickle', required=True)

    args = parser.parse_args()
    root_folder = args.root_folder
                        
    compare_models(root_folder)

