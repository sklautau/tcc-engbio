import json
import tensorflow as tf
import tensorflow_hub as hub
from keras.preprocessing.image import ImageDataGenerator
import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from keras.callbacks import EarlyStopping
import seaborn as sn
import os
import sys
import shutil
from sklearn import metrics

plt.close("all")


img_size = 224
model_url = 'https://tfhub.dev/google/efficientnet/b0/feature-vector/1'
custom_objects = {'KerasLayer': hub.KerasLayer(model_url, input_shape=(img_size, img_size, 3), trainable=False)}

trained_model = tf.keras.models.load_model('D:/sofia/ufpa/tcc/outputs/albumentations/effnetb0_augment_teste2_C1_1-18k_output/model_effnetb0_augment_testeC2_1-18k_.h5', custom_objects=custom_objects)
print("Just got the trained model ", trained_model)

# from https://stackoverflow.com/questions/44267074/adding-metrics-to-existing-model-in-keras
#new_metrics = [
      #keras.metrics.TruePositives(name='tp'),
      #keras.metrics.FalsePositives(name='fp'),
      #keras.metrics.TrueNegatives(name='tn'),
      #keras.metrics.FalseNegatives(name='fn'), 
      #keras.metrics.BinaryAccuracy(name='accuracy'),
      #keras.metrics.Precision(name='precision'),
      #keras.metrics.Recall(name='recall'),
      #keras.metrics.AUC(name='auc'),
      #keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
#]

#trained_model.compile(optimizer=trained_model.optimizer,
#                        loss=trained_model.loss,
#                        metrics=trained_model.metrics+new_metrics)

image_size = (img_size,img_size)
batch_size = 16
test_folder = 'D:/sofia/ufpa/tcc/data_intofolders_balanced/test_ratio1-3/'
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
#test_loss, test_accuracy = trained_model.evaluate(test_generator)
#print('Test loss:', test_loss)
#print('Test accuracy:', test_accuracy)

evaluate_results = trained_model.evaluate(test_generator, verbose=0)
print(evaluate_results)
for name, value in zip(trained_model.metrics_names, evaluate_results):
  print(name, ': ', value)
print()

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

fpr, tpr, thresholds = metrics.roc_curve(true_labels, predictions, pos_label=1)
auc = metrics.auc(fpr, tpr)
print('AUC:', auc)

recall = recall_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels)
precis = precision_score(true_labels, pred_labels)


print('Recall: ', recall)
print('Precision: ', precis)
print('F1-score: ', f1)

#rec = tf.keras.metrics.Recall()
#rec.update_state(true_labels, predictions)
#print('Final recall result: ', rec.result().numpy())  # Final result

# Compute confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
print('conf matrix\n', cm)
#%%
# save json
dictionary = dict()
dictionary['recall'] = recall
dictionary['precision'] = precis
dictionary['auc'] = auc
dictionary['f1'] = f1
#dictionary['cm'] = tuple(cm)

train_acc = [0.6166868209838867, 0.6541717052459717, 0.667067289352417, 0.6977025270462036, 0.7037484645843506, 0.7073760628700256]

dictionary['train_acc'] = train_acc

with open("test.json", "w") as outfile:
    json.dump(dictionary, outfile)
print('just saved file:', outfile)    

#%%



#%%

# Write accuracies and losses to a text file
base_name = "to_do"
output_dir = ".ls"
file_name = 'metrics_' + base_name + '.txt'
with open(os.path.join(output_dir, file_name), 'w') as f:
    f.write('recall\n')
    f.write(str(recall))
    f.write('\nprecision\n')
    f.write(str(precis))
    f.write('\nauc\n')
    f.write(str(auc))
    f.write('\nf1\n')
    f.write(str(f1))
    