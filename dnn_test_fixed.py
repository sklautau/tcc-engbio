import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from keras.preprocessing.image import ImageDataGenerator

# Load the CSV files
# attention: all the JPG files must be in the same directory as the CSV files
# csv_train = 'D:/sofia/ufpa/tcc/dataset_updated/train_data.csv'
# df_train = pd.read_csv(csv_train, encoding="utf-16", dtype=str,)

# csv_test = 'D:/sofia/ufpa/tcc/dataset_updated/test_data.csv'
# df_test = pd.read_csv(csv_test, encoding="utf-16", dtype=str)

# csv_val = 'D:/sofia/ufpa/tcc/dataset_updated/val_data.csv'
# df_val = pd.read_csv(csv_val, encoding="utf-16", dtype=str)

csv_train = 'D:/sofia/ufpa/tcc/dataset_updated/train_data.csv'
df_train = pd.read_csv(csv_train, dtype=str)
#df_train = pd.read_csv(csv_train, encoding="latin1", dtype=str,)

csv_test = 'D:/sofia/ufpa/tcc/dataset_updated/test_data.csv'
df_test = pd.read_csv(csv_test, dtype=str)

csv_val = 'D:/sofia/ufpa/tcc/dataset_updated/val_data.csv'
df_val = pd.read_csv(csv_val, dtype=str)

df_val['target'] = df_val['target'].astype(str) 
df_test['target'] = df_test['target'].astype(str) 
df_train['target'] = df_train['target'].astype(str) 

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
    model.add(Dense(64, activation='relu'))
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

# Create an instance of the ImageDataGenerator class to load the images and the labels
datagen = ImageDataGenerator(rescale=1./255)

# Create a generator for the training data
train_generator = datagen.flow_from_dataframe(
    dataframe=df_train,
    directory='D:/sofia/ufpa/tcc/dataset_updated/train/',
    validate_filenames=False,
    x_col='image_name', #change to own column name
    y_col='target',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_generator = datagen.flow_from_dataframe(
    dataframe=df_test,
    directory='D:/sofia/ufpa/tcc/dataset_updated/test/',
    validate_filenames=False,
    x_col='image_name', #change to own column name
    y_col='target',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

val_generator = datagen.flow_from_dataframe(
    dataframe=df_val,
    directory='D:/sofia/ufpa/tcc/dataset_updated/val/',
    validate_filenames=False,
    x_col='image_name', #change to own column name
    y_col='target',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


weight_for_0 = 0.51
weight_for_1 = 28.63
class_weight = {0: weight_for_0, 1: weight_for_1}

# Train the model
model.fit(
    train_generator,
    validation_data = val_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    batch_size=batch_size,
    class_weight=class_weight)

model.summary()

test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)

# Save the model
model.save('model_test.h5')
