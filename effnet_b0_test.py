import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model_url = 'https://tfhub.dev/google/efficientnet/b0/feature-vector/1'

model = hub.KerasLayer(model_url, trainable=False)
#print(model.summary())

csv_train = 'D:/sofia/ufpa/tcc/dataset_updated/train_data.csv'
df_train = pd.read_csv(csv_train, dtype=str)

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


# Use the pre-trained model as a feature extractor
train_data = train_generator.map(lambda x, y: (model(x), y))
val_data = val_generator.map(lambda x, y: (model(x), y))


