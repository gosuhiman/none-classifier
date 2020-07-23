import os
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from util import plot_images

# Paths
data_set_url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
model_path = 'models/'

path_to_zip = keras.utils.get_file('cats_and_dogs.zip', origin=data_set_url, extract=True)

data_path = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
train_path = os.path.join(data_path, 'train')
validation_path = os.path.join(data_path, 'validation')

image_size = (224, 224)
batch_size = 10
epochs = 15

train_data_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15
)
validation_data_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15
)

train_directory_iterator = train_data_generator.flow_from_directory(
    directory=train_path,
    target_size=image_size,
    class_mode='binary',
    batch_size=batch_size
)
validation_directory_iterator = validation_data_generator.flow_from_directory(
    directory=validation_path,
    target_size=image_size,
    class_mode='binary',
    batch_size=batch_size
)

sample_training_images, _ = next(train_directory_iterator)
plot_images(sample_training_images[:5])

vgg19_model = keras.applications.vgg19.VGG19()

model = Sequential()
for layer in vgg19_model.layers[:-1]:
    model.add(layer)

for layer in model.layers:
    layer.trainable = False

model.add(Dense(1, activation='softmax'))
model.summary()
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(
    train_directory_iterator,
    validation_data=validation_directory_iterator,
    steps_per_epoch=train_directory_iterator.samples / epochs,
    validation_steps=validation_directory_iterator.samples / epochs,
    epochs=epochs
)

if not os.path.exists(model_path):
    os.makedirs(model_path)

model.save(model_path + 'model.h5')
