# cat-finder

### Data augmentation

```
train_data_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True,
    zoom_range=0.5,
    shear_range=0.15
)
```

```
loss: 0.2101 - accuracy: 0.9100 - val_loss: 0.1524 - val_accuracy: 0.9400
```
