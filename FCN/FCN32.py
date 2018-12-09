'''image segmentation FCN32'''
from keras.models import *
from keras.layers import *
import os
file_path=os.path.dirname(os.path.abspath(__file__))

VGG_weight_path='...'

def FCN32(n_classes,input_height=416,input_width=608,vgg_level=3):

    assert input_height%32==0
    assert input_width%32==0

    img_input=Input(shape=(input_height,input_width,3))

    #block 1
    x=Conv2D(64,(3,3),activation='relu',padding='same',name='block1_conv1')(img_input)
    x=Conv2D(64,(3,3),activation='relu',padding='same',name='block1_conv2')(x)
    x=MaxPooling2D((2,2),strides=(2,2),name='block1_maxpool')(x)
    f1=x
    #block 2
    x=Conv2D(128,(3,3),activation='relu',padding='same',name='block2_conv1')(x)
    x=Conv2D(128,(3,3),activation='relu',padding='same',name='block2_conv2')(x)
    x=MaxPooling2D((2,2),strides=(2,2),name='block2_maxpool')(x)
    f2=x
    #block 3
    x=Conv2D(256,(3,3),activation='relu',padding='same',name='block3_conv1')(x)
    x=Conv2D(256,(3,3),activation='relu',padding='same',name='block3_conv2')(x)
    x=Conv2D(256,(3,3),activation='relu',padding='same',name='block3_conv3')(x)
    x=MaxPooling2D((2,2),strides=(2,2),name='block3_maxpool')(x)
    f3=x
    #block 4
    x=Conv2D(512,(3,3),activation='relu',padding='same',name='block4_conv1')(x)
    x=Conv2D(512,(3,3),activation='relu',padding='same',name='block4_conv2')(x)
    x=Conv2D(512,(3,3),activation='relu',padding='same',name='block4_conv3')(x)
    x=MaxPooling2D((2,2),strides=(2,2),name='block4_maxpool')(x)
    f4=x
    #block 5
    x=Conv2D(512,(3,3),activation='relu',padding='same',name='block5_conv1')(x)
    x=Conv2D(512,(3,3),activation='relu',padding='same',name='block5_conv2')(x)
    x=Conv2D(512,(3,3),activation='relu',padding='same',name='block5_conv3')(x)
    x=MaxPooling2D((2,2),strides=(2,2),name='block5_maxpool')(x)
    f5=x

    x=Flatten(name='flatten')(x)
    x=Dense(4096,activation='relu',name='fc1')(x)
    x=Dense(4096,activation='relu',name='fc2')(x)
    x=Dense(1000,activation='softmax',name='predictions')(x)

    vgg=Model(img_input,x)
    vgg.load_weights(VGG_weight_path)

    o=f5

    o=Conv2D(4096,(7,7),activation='relu',padding='same')(o)
    o=Dropout(0.5)
    o=Conv2D(4096,(1,1),activation='relu',padding='same')(o)
    o=Dropout(0.5)(o)

    o=Conv2D(n_classes,(1,1),kernel_initializer='he_normal')(o)
    o=Conv2DTranspose(n_classes,kernel_size=(64,64),strides=(32,32),use_bias=False)(o)
    o_shape=Model(img_input,o).output_shape

    outputHeight=o_shape[1]
    outputWidth=o_shape[2]

    print(o_shape) 

    o=Reshape((outputHeight*outputWidth,-1))(o)
    o=Permute((2,1))(o)
    o=Activation('softmax')(o)

    model=Model(img_input,o)
    model.outputWidth=outputWidth
    model.outputHeight=outputHeight 

    return model

if __name__=='__main__':
    m=FCN32(101)
    from keras.utils import plot_model
    plot_model(m,show_shapes=True,to_file='model.png')