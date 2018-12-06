'''
Fine-tuning 97% test acc
    1.Add your custom network on top of an already-trained base network
    2.Freeze the base network
    3.Train the part you added
    4.Unfreeze some layers in the base network
    5.Jointly train both these layers and the part you added
'''
from keras import models
from keras import layers
from keras.applications import VGG16
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import os

base_dir='E:/DataSet/catsanddogs_small'

train_dir=os.path.join(base_dir,'train')
test_dir=os.path.join(base_dir,'test')
validation_dir=os.path.join(base_dir,'validation')

conv_base=VGG16(weights='imagenet',
                include_top=False,
                input_shape=(150,150,3))
conv_base.trainable=False #Freeze VGG net

#Adding a densely connected classifier on top of the convolutional base
model=models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

#Training the model end to end with a frozen convolutional base
train_datagen=ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
test_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(
        train_dir,
        target_size=(150,150),
        batch_size=20,
        class_mode='binary')
validation_generator=test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150,150),
        batch_size=20,
        class_mode='binary')

model.compile(loss='binary_crossentropy',
        optimizer=optimizers.RMSprop(lr=2e-5),
        metrics=['acc'])

history=model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50)


'''PART TWO'''
#Freezing all layers up to a specific one
conv_base.trainable=True

set_trainable=False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable=True
    if set_trainable:
        layer.trainable=True
    else:
        layer.trainable=False

#Fine-tuning the model
model.compile(loss='binary_crossentropy',
                optimizer=optimizers.RMSprop(lr=1e-5),
                metrics=['acc'])
history=model.fit_generator(
                train_generator,
                steps_per_epoch=100,
                epochs=100,
                validation_data=validation_generator,
                validation_steps=50)
