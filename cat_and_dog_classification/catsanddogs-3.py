''''Fast Feature Extraction without Data Augmentation'''
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
import os
import numpy as np

conv_base=VGG16(weights='imagenet',
                include_top=False,
                input_shape=(150,150,3))

conv_base.summary()

base_dir='E:/DataSet/catsanddogs_small'

train_dir=os.path.join(base_dir,'train')
test_dir=os.path.join(base_dir,'test')
validation_dir=os.path.join(base_dir,'validation')

datagen=ImageDataGenerator(rescale=1./255)
batch_size=20

#Extracting features using the pretrained convolutional network
def extract_feature(directory,sample_count):
    features=np.zeros(shape=(sample_count,4,4,512))
    labels=np.zeros(shape=(sample_count))
    generator=datagen.flow_from_directory(
        directory,
        target_size=(150,150),
        batch_size=batch_size,
        class_mode='binary')
    i=0
    for inputs_batch,labels_batch in generator:
        features_batch=conv_base.predict(inputs_batch)
        features[i*batch_size:(i+1)*batch_size]=features_batch
        labels[i*batch_size:(i+1)*batch_size]=labels_batch
        i+=1
        if i*batch_size >= sample_count:
            break
    return features,labels

train_features,train_labels=extract_feature(train_dir,2000)
validation_features,validation_labels=extract_feature(validation_dir,1000)
test_features,test_labels=extract_feature(test_dir,1000)

train_features=np.reshape(train_features,(2000,4*4*512))
validation_features=np.reshape(validation_features,(1000,4*4*512))
test_features=np.reshape(test_features,(1000,4*4*512))

#Defining and training the densely connected classifier
model=models.Sequential()
model.add(layers.Dense(256,activation='relu',input_dim=4*4*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                loss='binary_crossentropy',
                metrics=['acc'])

history=model.fit(train_features,train_labels,
                epochs=30,
                batch_size=20,
                validation_data=(validation_features,validation_labels))

#Plotting the results
import matplotlib.pyplot as plt 
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Traning acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and Validation acc')
plt.legend()

plt.figure()

plt.plot(epochs,loss,'go',label='Training loss')
plt.plot(epochs,val_loss,'g',label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()