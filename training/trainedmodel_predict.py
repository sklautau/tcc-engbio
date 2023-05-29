# this script uses a trained model with effnet bX to test its predictions

import tensorflow as tf
import tensorflow_hub as hub
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import os
import sys
import shutil

plt.close("all")


#%% CHECK ALL VARIABLES AND REPLACE VALUES AS NEEDED
#--------------------------------------------------------------------------
img_size = 224
model_url = 'https://tfhub.dev/google/efficientnet/b1/feature-vector/1'
use_augment = True
type_augment = "Albumentations"
test_id = str(2) #test number
data_id = "1-18k-test50-50" #ratio
effnet_model = str(1) 

extractor = hub.KerasLayer(model_url, input_shape=(img_size, img_size, 3))

custom_objects = {'KerasLayer': hub.KerasLayer(extractor, trainable=False)}

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

trained_model = tf.keras.models.load_model('D:/sofia/ufpa/tcc/outputs/albumentations/effnetb0_augment_testeC1_1-18k_output/model_effnetb0_augment_testeC1_1-18k_.h5', custom_objects=custom_objects)
print("Just got the trained model ", trained_model)

image_size = (224,224)
batch_size = 16
test_folder = 'D:/sofia/ufpa/tcc/data_intofolders_balanced/test_50percent_teste4/'
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_folder,
    # test_folder,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)
# Evaluating the model on  the test set
test_loss, test_accuracy = trained_model.evaluate(test_generator)
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)

predictions = trained_model.predict(test_generator)

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
    f.write('\ntest_acc\n')
    f.write(str(test_accuracy))
    

