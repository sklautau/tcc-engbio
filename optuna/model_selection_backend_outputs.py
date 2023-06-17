'''
This version does not use generators but the pre-calculated outputs of the backend, to save time.

Performs hyperparameter search using Optuna (https://github.com/optuna/optuna-examples/blob/main/keras/keras_simple.py)
Uses datagen.flow_from_dataframe instead of datagen.flow_from_directory.
In this example, we optimize the validation accuracy using
Keras. We optimize hyperparameters such as the filter and kernel size, and layer activation.

References:
https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html#pruning
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
ID = 31 # identifier for this simulation - use 1000 examples (unbalanced) instead of 648 - only 3 parameters
EPOCHS = 100 # maximum number of epochs
#IMAGESIZE = (240, 240)      # Define the input shape of the images
INPUTSHAPE = (1280,) # (240, 240, 3)  # NN input
#BEST_MODEL = None # Best NN model 
#CURRENT_MODEL = None
VERBOSITY_LEVEL = 0 #use 1 to see the progress bar when training and testing

#folder with 3 files storing pre-computed backend outputs
#INPUT_DIR = '../backend_output/effnet_N1000/' 
INPUT_DIR = '../backend_output/efficientnet_v2_imagenet21k_ft1k_xl_N1000/'

#Important: output folder
OUTPUT_DIR = '../outputs/optuna_outputs/id_' + str(ID) + '/' #os.path.join('../outputs/unbalanced/id_' + str(simulation_ID), base_name)        
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print("Created folder ", OUTPUT_DIR)

# not working! CURRENT_MODEL is None
def save_best_model_callback(study, trial):
    global BEST_MODEL, OUTPUT_DIR
    best_model_name = "optuna_best_model" # do not use .h5 extension to save in modern format
    best_model_name = os.path.join(OUTPUT_DIR, best_model_name)
    if study.best_trial == trial:
        #BEST_MODEL = CURRENT_MODEL
        print("Saving best model ", best_model_name, "...")
        #BEST_MODEL.save(best_model_name)

def read_three_datasets():
    file_name = os.path.join(INPUT_DIR, "validation.pickle")
    val_data = read_dataset(file_name)
    file_name = os.path.join(INPUT_DIR, "train.pickle")
    train_data = read_dataset(file_name)
    file_name = os.path.join(INPUT_DIR, "test.pickle")
    test_data = read_dataset(file_name)

    return train_data, test_data, val_data

def read_dataset(file_name):
    file = os.path.join(file_name)
    with open(file, "rb") as file_pi:
        examples = pickle.load(file_pi)
    return examples

def objective(trial): # uses effnet
    # Clear clutter from previous Keras session graphs.
    clear_session()

    num_output_neurons = 1

    train_data, test_data, val_data = read_three_datasets()    
    # train_generator, validation_generator, test_generator = get_data_generators(num_desired_negative_train_examples, batch_size)
    #test_generator = None #not used here

    # Define the CNN model
    model = Sequential()
    #model.add(Input(input_shape=INPUTSHAPE))

    # Load the respective EfficientNet model but exclude the classification layers
    #trainable = False
    #model_url = 'https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b1/feature_vector/2'
    #extractor = hub.KerasLayer(model_url, input_shape=INPUTSHAPE, trainable=trainable)
    
    #Optuna parameters
    batch_size = 10 #trial.suggest_int("batch_size", 1, 15) 
    num_dense_layers = 1 #trial.suggest_int("num_dense_layers", 1, 2) # number of layers
    num_neurons_per_layer = np.zeros(num_dense_layers, dtype=np.int64)
    num_neurons_per_layer[0] = trial.suggest_int("neurons_L1", 50, 300)
    for i in range(num_dense_layers-1):
        # force number of neurons to not increase
        num_neurons_per_layer[i+1] = trial.suggest_int("neurons_L{}".format(i+2), num_neurons_per_layer[i]//4, num_neurons_per_layer[i])
    dropout_rate = trial.suggest_float("dropout", 0.5, 0.99)
    use_batch_normalization=False #trial.suggest_categorical("batch_nor", [True, False])
    use_regularizers= False #trial.suggest_categorical("regul", [True, False])
    if use_regularizers:
        l1_weight = trial.suggest_categorical("l1_weight", [0, 1e-5, 1e-4, 1e-3, 1e-2])
        l2_weight = trial.suggest_categorical("l2_weight", [0, 1e-5, 1e-4, 1e-3, 1e-2])
    activation= trial.suggest_categorical("activation", ["elu", "swish"])
    # We compile our model with a sampled learning rate.
    learning_rate = trial.suggest_float("lea_rate", 1e-5, 1e-1, log=True)

    print("num_neurons_per_layer =", num_neurons_per_layer)
    #first layer
    if use_regularizers:
            model.add(
                Dense(
                    # Define the number of neurons for this layer
                    num_neurons_per_layer[0],
                    activation= activation,
                    input_shape=INPUTSHAPE,
                    kernel_regularizer=regularizers.L1L2(l1=l1_weight, l2=l2_weight),
                    bias_regularizer=regularizers.L2(l2_weight),
                    activity_regularizer=regularizers.L2(l2_weight)                
                )
            )
    else:
            model.add(
                Dense(
                    # Define the number of neurons for this layer
                    num_neurons_per_layer[0],
                    activation= activation,
                    input_shape=INPUTSHAPE
                )
            )

    for i in range(1,num_dense_layers):
        if use_regularizers:
            model.add(
                Dense(
                    # Define the number of neurons for this layer
                    num_neurons_per_layer[i],
                    activation= activation,
                    input_shape=INPUTSHAPE,
                    kernel_regularizer=regularizers.L1L2(l1=l1_weight, l2=l2_weight),
                    bias_regularizer=regularizers.L2(l2_weight),
                    activity_regularizer=regularizers.L2(l2_weight)                
                )
            )
        else:
            model.add(
                Dense(
                    # Define the number of neurons for this layer
                    num_neurons_per_layer[i],
                    activation= activation,
                    input_shape=INPUTSHAPE
                )
        )
        # first and most important rule is: don't place a BatchNormalization after a Dropout
        # https://stackoverflow.com/questions/59634780/correct-order-for-spatialdropout2d-batchnormalization-and-activation-function
        if use_batch_normalization:
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    if use_regularizers:
        model.add(Dense(num_output_neurons, 
                        activation="sigmoid",
                        kernel_regularizer=regularizers.L1L2(l1=l1_weight, l2=l2_weight),
                        bias_regularizer=regularizers.L2(l2_weight),
                        activity_regularizer=regularizers.L2(l2_weight)                
        ))
    else:
        model.add(Dense(num_output_neurons, activation="sigmoid"))

    model.summary()


    # Define the metric for callbacks and Optuna
    if True:
        metric_to_monitor = ('val_accuracy',) #trial.suggest_categorical("metric_to_monitor", ['val_accuracy', 'val_auc']),
    else:
        metric_to_monitor = ('val_auc',)
    metric_mode = 'max'
    early_stopping = EarlyStopping(monitor=metric_to_monitor[0], patience=3, mode=metric_mode)
    #early_stopping = EarlyStopping(monitor='val_auc', patience=5)

    #look at https://www.tensorflow.org/guide/keras/serialization_and_saving
    #do not use HDF5 (.h5 extension)
    #best_model_name = 'best_model_' + base_name + '.h5'
    best_model_name = 'optuna_best_model_' + str(trial.number)
    best_model_name = os.path.join(OUTPUT_DIR, best_model_name)
    best_model_save = ModelCheckpoint(best_model_name, save_best_only=True, monitor=metric_to_monitor[0], mode=metric_mode)

    reduce_lr_loss = ReduceLROnPlateau(monitor=metric_to_monitor[0], factor=0.5, patience=3, verbose=VERBOSITY_LEVEL, min_delta=1e-4, mode=metric_mode)
    # Define Tensorboard as a Keras callback
    tensorboard = TensorBoard(
    log_dir= '.\logoptuna',
    #log_dir= '.\logs',
    histogram_freq=1,
    write_images=True
    )

    print("")
    print("------------------------------------------------------------")
    print("------------------------------------------------------------")
    print("  Hyperparameters of Optuna trial # ", trial.number)
    print("------------------------------------------------------------")
    print("------------------------------------------------------------")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    model.compile(
        loss="binary_crossentropy",
        #optimizer=RMSprop(learning_rate=learning_rate),
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy", tf.keras.metrics.AUC()] #always use both metrics, and choose one to guide Optuna
    )

    
    num_train = len(train_data[1])
    # Training the model
    history = model.fit(
        x=train_data[0], 
        y=train_data[1], 
        steps_per_epoch=num_train // batch_size,
        epochs=EPOCHS,
        validation_data=(val_data[0], val_data[1]),
        verbose=VERBOSITY_LEVEL,
        #callbacks=[early_stopping,reduce_lr_loss, tensorboard]
        #callbacks=[TFKerasPruningCallback(trial, metric_to_monitor), early_stopping]
        #callbacks=[early_stopping, best_model_save, reduce_lr_loss]
        callbacks=[early_stopping, best_model_save, reduce_lr_loss, TFKerasPruningCallback(trial, metric_to_monitor[0])]
    )
    #CURRENT_MODEL = tf.keras.models.clone_model(model)

    # add to history
    history.history['num_desired_train_examples'] = num_train

    # https://stackoverflow.com/questions/41061457/keras-how-to-save-the-training-history-attribute-of-the-history-object
    pickle_file_path = os.path.join(OUTPUT_DIR, 'optuna_best_model_' + str(trial.number), 'trainHistoryDict.pickle')
    with open(pickle_file_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    
    # Evaluate the model accuracy on the validation set.
    # score = model.evaluate(x_valid, y_valid, verbose=0)

    if True:
        #train data
        print('Train loss:', history.history['loss'][-1])
        print('Train accuracy:', history.history['accuracy'][-1])
        print('Train AUC:', history.history['auc'][-1])

    if True:  # test data cannot be used in model selection. This is just sanity check
        test_loss, test_accuracy, test_auc = model.evaluate(x=test_data[0], y=test_data[1], verbose=VERBOSITY_LEVEL)
        print('Test loss:', test_loss)
        print('Test accuracy:', test_accuracy)
        print('Test AUC:', test_auc)

    # Evaluate the model accuracy on the validation set.
    #val_loss, val_accuracy, val_auc = model.evaluate(validation_generator, verbose=VERBOSITY_LEVEL)
    val_accuracy = history.history['val_accuracy'][-1]
    val_auc = history.history['val_auc'][-1]
    #print('Val loss:', val_loss)
    #print('Val accuracy:', val_accuracy)
    #print('Val AUC:', val_auc)
    #avoid above by using pre-calculated:
    print('Val loss:', history.history['val_loss'][-1])
    print('Val accuracy:', val_accuracy)
    print('Val AUC:', val_auc)

    # Optuna needs to use the same metric for all evaluations (it could be val_accuracy or val_auc but one cannot change it for each trial)

    if metric_to_monitor[0] == 'val_accuracy': #trial.suggest_categorical("metric_to_monitor", ['val_accuracy', 'val_auc']),
        return val_accuracy
    elif metric_to_monitor[0] == 'val_auc':
        return val_auc
    else:
        raise Exception("Metric must be val_auc or val_accuracy")
        


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
    print("Model selection using pre-computed backend outputs")

    #copy script
    copied_script = os.path.join(OUTPUT_DIR, os.path.basename(sys.argv[0]))
    shutil.copy2(sys.argv[0], copied_script)
    print("Just copied current script as file", copied_script)

    #study = optuna.create_study(direction="maximize")
    study = optuna.create_study(direction="maximize",
                                storage="sqlite:///optuna_db.sqlite3",  # Specify the storage URL here.
                                study_name='ID_' + str(ID),
                                sampler=optuna.samplers.TPESampler(), 
                                pruner=optuna.pruners.HyperbandPruner())
    #study.optimize(objective, n_trials=100)
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])    
    study.optimize(objective, n_trials=250) #, callbacks=[save_best_model_callback]) #, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    trial = study.best_trial
    print("Best trial is #", trial.number)
    print("  Value: {}".format(trial.value))

    print("  Hyperparameters: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    with open(os.path.join(OUTPUT_DIR, 'best_optuna_trial.txt'), 'w') as f:
        f.write("Best trial is #" + str(trial.number))
        f.write('\n')
        f.write("  Value: {}".format(trial.value))
        f.write('\n')
        f.write("  Hyperparameters: ")
        f.write('\n')
        for key, value in trial.params.items():
            f.write("    {}: {}".format(key, value))
            f.write('\n')

    pickle_file_path = os.path.join(OUTPUT_DIR, 'study.pickle')
    with open(pickle_file_path, 'wb') as file_pi:
        pickle.dump(study, file_pi)
    print("Wrote", pickle_file_path)

    #https://optuna.readthedocs.io/en/stable/reference/visualization/generated/optuna.visualization.plot_optimization_history.html
    plt.close("all")
    plt.figure()
    # Using optuna.visualization.plot_optimization_history(study) invokes the other Optuna's backend. To use matplotlib, use:
    optuna.visualization.matplotlib.plot_optimization_history(study) #optimization history
    # Save the figure to a file (e.g., "optimization_history.png")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'optuna_optimization_history.png'))
    plt.close("all")
    #fig.show()
    optuna.visualization.matplotlib.plot_intermediate_values(study) #Visualize the loss curves of the trials
    plt.savefig(os.path.join(OUTPUT_DIR, 'optuna_loss_curves.png') )
    #fig.show()
    plt.close("all")
    optuna.visualization.matplotlib.plot_contour(study) #Parameter contour plots 
    plt.savefig(os.path.join(OUTPUT_DIR, 'optuna_contour_plots.png') )
    #fig.show()
    plt.close("all")
    optuna.visualization.matplotlib.plot_param_importances(study) # parameter importance plot
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'optuna_parameter_importance.png') )
    #fig.show()

