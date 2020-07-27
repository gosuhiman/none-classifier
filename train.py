import os
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from util import plot_images, plot_accuracy_and_loss

# Paths
data_set_url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
model_path = 'models/'

path_to_zip = keras.utils.get_file('cats_and_dogs.zip', origin=data_set_url, extract=True)

data_path = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
train_path = os.path.join(data_path, 'train')
validation_path = os.path.join(data_path, 'validation')
train_cats_path = os.path.join(train_path, 'cats')
train_dogs_path = os.path.join(train_path, 'dogs')
validation_cats_path = os.path.join(validation_path, 'cats')
validation_dogs_path = os.path.join(validation_path, 'dogs')

num_cats_train = len(os.listdir(train_cats_path))
num_dogs_train = len(os.listdir(train_dogs_path))
num_cats_validation = len(os.listdir(validation_cats_path))
num_dogs_validation = len(os.listdir(validation_dogs_path))
total_train = num_cats_train + num_dogs_train
total_validation = num_cats_validation + num_dogs_validation

print('Total training cat images:', num_cats_train)
print('Total training dog images:', num_dogs_train)
print('Total validation cat images:', num_cats_validation)
print('Total validation dog images:', num_dogs_validation)
print("-----")
print("Total training images:", total_train)
print("Total validation images:", total_validation)

image_size = (224, 224)
classes = ['cats', 'dogs']
batch_size = 16
epochs = 15

train_data_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True,
    zoom_range=0.5,
    shear_range=0.15
)
validation_data_generator = ImageDataGenerator(
    rescale=1./255
)

train_directory_iterator = train_data_generator.flow_from_directory(
    directory=train_path,
    target_size=image_size,
    classes=classes,
    batch_size=batch_size
)
validation_directory_iterator = validation_data_generator.flow_from_directory(
    directory=validation_path,
    target_size=image_size,
    classes=classes,
    batch_size=batch_size
)

sample_training_images, _ = next(train_directory_iterator)
plot_images(sample_training_images[:5])

vgg19_model = keras.applications.vgg19.VGG19()

model = Sequential()
for layer in vgg19_model.layers[:-1]:
    model.add(layer)

for layer in model.layers[:-7]:
    layer.trainable = False

model.add(Dense(2, activation='softmax'))
model.summary()
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

history = model.fit(
    train_directory_iterator,
    validation_data=validation_directory_iterator,
    steps_per_epoch=total_train / batch_size,
    validation_steps=total_validation / batch_size,
    epochs=epochs
)

if not os.path.exists(model_path):
    os.makedirs(model_path)

model.save(model_path + 'model.h5')

plot_accuracy_and_loss(history, epochs)
