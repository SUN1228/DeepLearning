#DenseNet
from keras.models import Model
from keras.layers.core import Dense,Dropout,Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input,Concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as k 

def conv_factory(x,concat_axis,nb_filter,dropout_rate=None,weight_decay=1E-4):
    '''BN->ReLU->3x3Conv'''
    x=BatchNormalization(axis=concat_axis,
                        gamma_regularizer=l2(weight_decay),
                        beta_regularizer=l2(weight_decay))(x)
    x=Activation('relu')(x)
    x=Conv2D(nb_filter,(3,3),
                kernel_initializer='he_uniform',
                padding='same',
                use_bias=False,
                kenel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x=Dropout(dropout_rate)(x)
    
    return x

def transition(x,concat_axis,nb_filter,dropout_rate=None,weight_decay=1E-4):
    '''BN->ReLU->1x1Conv'''
    x=BatchNormalization(axis=concat_axis,
                        gamma_regularizer=l2(weight_decay),
                        beta_regularizer=l2(weight_decay))(x)
    x=Activation('relu')(x)
    x=Conv2D(nb_filter,(1,1),
                kernel_initializer='he_uniform',
                padding='same',
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x=Dropout(dropout_rate)(x)
    x=AveragePooling2D((2,2),strides=(2,2))(x)

    return x

def denseblock(x,concat_axis,nb_layers,nb_filter,growth_rate,dropout_rate=None,weight_decay=1E-4):
    list_feat=[x]

    for i in range(nb_layers):
        x=conv_factory(x,concat_axis,growth_rate,dropout_rate,weight_decay)
        list_feat.append(x)
        x=Concatenate(axis=concat_axis)(list_feat)
        nb_filter+=growth_rate

    return x,nb_filter

def DenseNet(nb_classes,img_dim,depth,nb_dense_block,growth_rate,nb_filter,dropout_rate=None,weight_decay=1E-4):
    if K.image_dim_orderding()=='th':
        concat_axis=1
    else K.image_dim_orderding()=='tf':
        concat_axis=-1
    
    model_input=Input(shape=img_dim)

    assert (depth-4)%3==0,"depth must be 3N+4"

    #layers in each dense block
    nb_layers=int((depth-4)/3)

    #Initial convolution
    x=Conv2D(nb_filter,(3,3),
                kernel_initializer='he_uniform',
                padding='same',
                name='initial_conv2d',
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(model_input)
    
    #add dense blocks
    for block_idx in range(nb_dense_block-1):
        x,nb_filter=denseblock(x,concat_axis,nb_layers,nb_filter,growth_rate,dropout_rate=dropout_rate,weight_decay=weight_decay)
        #add transition
        x=transition(x,nb_filter,dropout_rate=dropout_rate,weight_decay=weight_decay)
    
    #the last denseblock
    x,nb_filter=denseblock(x,concat_axis,nb_layers,nb_filter,nb_filter,growth_rate,dropout_rate=dropout_rate,weight_decay=weight_decay)
    x=BatchNormalization(axis=concat_axis,gamma_regularizer=l2(weight_decay),beta_regularizer=l2(weight_decay))(x)
    x=Activation('relu')(x)
    x=GlobalAveragePooling2D(data_format=K.image_data_format())(x)
    x=Dense(nb_classes,activation='softmax',kernel_regularizer=l2(weight_decay),bias_regularizer=l2(weight_decay))(x)

    densenet=Model(inputs=[model_input],output=[x],name='densenet')

    return densenet
