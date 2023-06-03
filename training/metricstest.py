import tensorflow as tf
import tensorflow_hub as hub
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score


plt.close("all")

img_size = 224
model_url = 'https://tfhub.dev/google/efficientnet/b3/feature-vector/1'


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
                  metrics=["accuracy",
                  # tf.keras.metrics.Precision(),
                  # tf.keras.metrics.Recall(),
                  # tf.keras.metrics.TrueNegatives(),
                  # tf.keras.metrics.TruePositives(),
                  
                  ])
    
    model.summary()
    
    return model


model = get_training_model(model_url) 
print("just got the model")

# Define the folders for train, validation, and test data
train_folder = 'D:/sofia/ufpa/tcc/data_intofolders/albumentations/train-1-C1/'
validation_folder = 'D:/sofia/ufpa/tcc/data_intofolders_balanced/val/'
test_folder = 'D:/sofia/ufpa/tcc/data_intofolders_balanced/test_ratio1-3/'

# Define the image size and other parameters
image_size = (img_size, img_size)
batch_size = 16
epochs = 3
num_classes = 2

# Loading and preprocessing the training, validation, and test data
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
 
train_generator = train_datagen.flow_from_directory(
    # train_folder,
    test_folder,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    # validation_folder,
    test_folder,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    # test_folder,
    test_folder,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Training the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    # callbacks=[early_stopping]
)

# test_loss, test_accuracy, test_precision, test_recall, test_tn, test_tp = model.evaluate(test_generator)

test_loss, test_accuracy = model.evaluate(test_generator)
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)

# Generate predictions --> labels: predicted and true
predictions = model.predict(test_generator)

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
