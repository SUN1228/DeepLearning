import tensorflow as tf 
#import tensorflow.contrib.slim as slim 
slim=tf.contrib.slim

def vgg_arg_scope(weight_decay=1E-4):
    with slim.arg_scope([slim.conv2d,slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d],padding='SAME') as arg_sc:
            return arg_sc

def vgg_a(inputs,
          num_classes=1000,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='vgg_a',
          fc_conv_padding='VALID',
          global_pool=False):
    '''VGG 11 Layers Version
    note:使用conv2d代替全连接层
    
    inputs: a tensor of size [batch_size,height,width,channels]
    num_classes: number of predited classes
    is_training: whether or not the model is being trained
    dropout_keep_prob: probability that activations are kept in the dropout
    spatial_squeeze: whether of not should squeeze the spatial dimensions of the outpus
    scope: scope of variables
    fc_conv_padding: 
    global_pool: 全局平均池化

    net: the output of the logits layer (if num_classes is a non-zero integer),
         or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
    '''
    with tf.variable_scope(scope,'vgg_a',[inputs]) as sc:
        end_points_colltction=sc.original_name_scope+'_end_points'
        with slim.arg_scope([slim.conv2d,slim.max_pool2d],
                            outputs_collections=end_points_colltction):
            net=slim.repeat(inputs,1,slim.conv2d,64,[3,3],scope='conv1')
            net=slim.max_pool2d(net,[2,2],scope='pool1')
            net=slim.repeat(net,1,slim.conv2d,128,[3,3],scope='conv2')
            net=slim.max_pool2d(net,[2,2],scope='pool2')
            net=slim.repeat(net,2,slim.conv2d,256,[3,3],scope='conv3')
            net=slim.max_pool2d(net,[2,2],scope='pool3')
            net=slim.repeat(net,2,slim.conv2d,512,[3,3],scope='conv4')
            net=slim.max_pool2d(net,[2,2],scope='pool4')
            net=slim.repeat(net,2,slim.conv2d,512,[3,3],scope='conv5')
            net=slim.max_pool2d(net,[2,2],scope='pool5')

            #use conv2d instead of fully connected layers
            net=slim.conv2d(net,4096,[7,7],padding=fc_conv_padding,scope='fc6')
            net=slim.dropout(net,dropout_keep_prob,is_training=is_training,scope='dropout6')
            net=slim.conv2d(net,4096,[1,1],scope='fc7')

            #convert end_points_collection into a end_point dict
            end_points=slim.utils.convert_collection_to_dict(end_points_colltction)
            if global_pool:
                net=tf.reduce_mean(net,[1,2],keep_dims=True,name='global_pool')#对长和宽两个维度计算平均值
                end_points['global_pool']=net
            if num_classes:
                net=slim.dropout(net,dropout_keep_prob,is_training=is_training,scope='dropout7')
                net=slim.conv2d(net,num_classes,[1,1],
                                activation_fn=None,
                                normalizer_fn=None,
                                scope='fc8')
            if spatial_squeeze:
                net=tf.squeeze(net,[1,2],name='fc8/suqeeze')
            end_points[sc.name+'fc8']=net
        return net,end_points
vgg_a.default_image_size=224
            


