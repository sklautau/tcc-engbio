#tutorial in part from https://wandb.ai/sayakpaul/efficientnet-tl/reports/Transfer-Learning-With-the-EfficientNet-Family-of-Models--Vmlldzo4OTg1Nw

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
img_size = 240
model_url = 'https://tfhub.dev/google/efficientnet/b1/feature-vector/1'
use_augment = True
type_augment = "Albumentations"
test_id = str(2) #test number
data_id = "1-18k-test50-50" #ratio
effnet_model = str(1) 
#--------------------------------------------------------------------------

base_name = 'effnetb' + effnet_model + '_augment_testeC' + test_id + '_' + data_id
output_dir = os.path.join('../outputs/albumentations/', base_name + '_output')

if use_augment:
    title_id = 'Augment - ' + type_augment + ' - test: ' + test_id + ' -' + ' ratio: ' + data_id
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
test_folder = 'D:/sofia/ufpa/tcc/data_intofolders_balanced/test_50percent_teste4/'

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
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)

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

# Evaluating the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)

# Generate predictions --> labels: predicted and true
predictions = model.predict(test_generator)

# defining the metrics
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Plot the metrics over epochs
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.legend()
plt.show()

metrics_name = 'metrics_' + base_name + '.png'
plt.title('Metrics - ' + title_id)
plt.savefig(os.path.join(output_dir, metrics_name))

# Convert predictions into values 0 or 1
N = len(predictions)
pred_labels = np.zeros( (N,) )
my_threshold = 0.5 # assume a threshold given predictions are in range [0, 1]
for i in range(N):
    if predictions[i] > my_threshold:
        pred_labels[i] = 1
    else:
        pred_labels[i] = 0
true_labels = test_generator.classes

# Compute confusion matrix
cm = confusion_matrix(true_labels, pred_labels)

# Plot confusion matrix with Seaborn
plt.figure()
sn.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion matrix - ' + title_id)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

#save the cm
cm_filename = 'cm_' + base_name + '.png'  
plt.savefig(os.path.join(output_dir, cm_filename))

# Write accuracies and losses to a text file
file_name = 'classification_output_' + base_name + '.txt'
with open(os.path.join(output_dir, file_name), 'w') as f:
    f.write('validation_acc\n')
    f.write(str(val_acc))
    f.write('\ntrain_acc\n')
    f.write(str(train_acc))
    f.write('\nvalidation_loss\n')
    f.write(str(val_loss))
    f.write('\ntrain_loss\n')
    f.write(str(train_loss))
    f.write('\ntest_acc\n')
    f.write(str(test_accuracy))
    




    
    