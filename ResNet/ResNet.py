'''Train a ResNet on the CIFAR10 dataset'''
import keras
from keras.layers import Dense,Conv2D,BatchNormalization,Activation
from keras.layers import AveragePooling2D,Input,Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K 
from keras.models import Model
from keras import regularizers
from keras.datasets import cifar10
import numpy as np 
import os

#Training parameters
batch_size=32
epochs=200
data_augmentation=True
num_classes=10

#Subtracting pixel mean impores accuracy
subtract_pixel_mean=True

#Model parameter
n=3

#Model version
version=1

#Computed depth
if version == 1:
    depth=n*6+2
elif version == 2:
    depth=n*9+2

model_type='ResNet%dv%d' % (depth,version)

#Load data
(x_train,y_train),(x_test,y_test)=cifar10.load_data()

#Input image dimensions
input_shape=x_train.shape[1:]

#Normalize data
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255

#Subtract pixel mean
if subtract_pixel_mean:
    x_train_mean=np.mean(x_train,axis=0) 
    x_train-=x_train_mean
    x_test_mean=np.mean(x_test,axis=0)
    x_test-=x_test_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

def lr_schedule(epoch):
    lr=1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('learning rate is ',lr)
    return lr

#Convert class vector to binary class matrices
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)

def resnet_layer(inputs,
                num_filters=16,
                kernel_size=3,
                strides=1,
                activation='relu',
                batch_normalization=True,
                conv_first=True):
    conv=Conv2D(num_filters,
                strides=strides,
                kernel_size=kernel_size,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(1e-4))
    x=inputs
    if conv_first:
        x=conv(x)
        if batch_normalization:
            x=BatchNormalization()(x)
        if activation is not None:
            x=Activation(activation)(x)
    else:
        if batch_normalization:
            x=BatchNormalization()(x)
        if activation is not None:
            x=Activation(activation)(x)
        x=conv(x)
    return x

def resnet_v1(input_shape,depth,num_classes=10):
    if(depth -2)%6!=0:
        raise ValueError('depth should be 6n+2')
    #Start model definition
    num_filters=16
    num_res_blocks=int((depth-2)/6)

    inputs=Input(shape=input_shape)
    x=resnet_layer(inputs=inputs)
    #Instantiate the stack of residual units
    for stack in range(n):
        for res_block in range(num_res_blocks):
            strides=1
            if stack > 0 and res_block == 0: #first layer
                strides=2 #downsample
            y = resnet_layer(inputs=x,
                            num_filters=num_filters,
                            strides=strides)
            y = resnet_layer(inputs=y,
                            num_filters=num_filters,
                            activation=None)
            if stack > 0 and res_block == 0:
                #linear projection residual shortcut connection
                #change dim
                x=resnet_layer(inputs=x,
                                num_filters=num_filters,
                                kernel_size=1,
                                strides=strides,
                                activation=None,
                                batch_normalization=False)
            x=keras.layers.add([x,y])
            x=Activation('relu')(x)
        num_filters*=2

    #Add classifier on top
    # v1 does not use BN after last shortcut connection-ReLU
    x=AveragePooling2D(pool_size=8)(x)
    y=Flatten()(x)
    outputs=Dense(num_classes,
                activation='softmax',
                kernel_initializer='he_normal')(y)
    #Instantiate model
    model=Model(inputs=inputs,outputs=outputs)
    return model    

model=resnet_v1(input_shape,depth,num_classes)

model.compile(loss='categorical_crossentropy',
            optimizer=Adam(lr=lr_schedule(0)),
            metrics=['acc'])
model.summary()

#Save model
save_dir=os.path.join(os.getcwd(),'saved_models')
model_name='cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath=os.path.join(save_dir,model_name)

#Prepare callbacks for model saving and for learning rate adjustment
checkpoint=ModelCheckpoint(filepath=filepath,
                            monitor='val_acc',
                            verbose=1,
                            save_best_only=True)

lr_scheduler=LearningRateScheduler(lr_schedule)

lr_reducer=ReduceLROnPlateau(factor=np.sqrt(0.1),
                            cooldown=0,
                            patience=5,
                            min_lr=0.5e-6)

callbacks=[checkpoint,lr_reducer,lr_scheduler]

#data augmentation
if not data_augmentation:
    model.fit(x_train,y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test,y_test),
            shuffle=True,
            callbacks=callbacks)
else:
    datagen=ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0
    )
    datagen.fit(x_train)

    #fit the model on the batches generated by datagen.flow()
    model.fit_generator(datagen.flow(x_train,y_train,batch_size=batch_size),
                        validation_data=(x_test,y_test),
                        epochs=epochs,
                        verbose=1,
                        workers=4,
                        callbacks=callbacks)
#score trained model
scores=model.evaluate(x_test,y_test,verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])