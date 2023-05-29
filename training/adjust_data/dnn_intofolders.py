# import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping


# Define the image dimensions
img_width, img_height = 224, 224

# Define the number of epochs
epochs = 10

# Define the batch size
batch_size = 4

# Define the number of classes
num_classes = 2

num_output_neurons = 1

# Define the number of filters for the first convolutional layer
num_filters = 32

# Define the size of the pooling area for max pooling
pool_size = (2, 2)

# Define the size of the convolutional kernel
kernel_size = (4, 4)

# Define the input shape of the images
input_shape = (img_width, img_height, 3)

# Define the CNN model
model = Sequential()
if True:
    model.add(Conv2D(num_filters, kernel_size, input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size))
    model.add(Conv2D(num_filters, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size))
    model.add(Conv2D(num_filters, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size))
    model.add(Conv2D(num_filters, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size))
    model.add(Conv2D(num_filters, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_output_neurons, activation='sigmoid'))
else:
    model.add(Input(shape=input_shape))
    model.add(Flatten())
    #model.add(Dense(12, activation='relu'))
    model.add(Dropout(0.1))
    #model.add(Dense(20, activation='relu'))
    model.add(Dense(num_output_neurons, activation='sigmoid'))
    
    
# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

#Define the folders for train, validation, and test data
# train_folder = 'D:/sofia/ufpa/tcc/data_intofolders/train/'
# validation_folder = 'D:/sofia/ufpa/tcc/data_intofolders/val/'
#test_folder = 'D:/sofia/ufpa/tcc/data_intofolders_balanced/test/'

## teste 5
train_folder = 'D:/sofia/ufpa/tcc/data_intofolders_balanced/train_ratio1-6/'
test_folder = 'D:/sofia/ufpa/tcc/data_intofolders_balanced/test_ratio1-6/'
validation_folder = 'D:/sofia/ufpa/tcc/data_intofolders_balanced/val_ratio1-6/'

# Define the image size and other parameters
image_size = (224, 224)
batch_size = 16
epochs = 100
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

val_generator = validation_datagen.flow_from_directory(
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
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3)

# Train the model
history = model.fit(
    train_generator,
    validation_data = val_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stopping])

# write the accuracy on the console
test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)

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

# Compute confusion matrix
cm = confusion_matrix(true_labels, pred_labels)

# Plot confusion matrix with Seaborn
plt.figure()
sn.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

#save the cm
cm_filename = 'cm_cnn_test1_ratio1-8.png'  
plt.savefig(cm_filename)

# Plot the training and validation metrics over epochs
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.legend()
plt.show()

#save the image
train_filename = 'cnn_test1_ratio1-8.png'  
plt.savefig(train_filename)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Write accuracies and losses to a text file
with open('cnn_test1_ratio1-8_classification_output.txt', 'w') as f:
    f.write('validation_acc\n')
    f.write(str(val_acc))
    f.write('\ntrain_acc\n')
    f.write(str(train_acc))
    f.write('\nvalidation_loss\n')
    f.write(str(val_loss))
    f.write('\ntrain_loss\n')
    f.write(str(train_loss))
    f.write('\ntest_acc\n')
    f.write(str(test_acc))


    

