'''inception v1'''
import tensorflow as tf 
import tensorflow.contrib.slim as slim 

def inception_arg_scope(weight_decay=4E-5,
                        ust_batch_norm=True,
                        batch_norm_decay=0.9997,
                        batch_norm_epsilon=0.001,
                        activation_fn=tf.nn.relu,
                        batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS)：
    '''Defines the default arg scope for inception module
    
    Args:
        weight_decay: weight dacay for regularizing
        use_baech_norm: whether or not use bn after every conv
        batch_norm_decay: decay for batch norm moving average
        batch_norm_epsilon: small float added to variance to avoid
                            dividing by zeros
        activation_fn: activation function for conv2d
        batch_norm_updatas_collections: collection for the update ops for bn
    
    Returns:
        arg_scope
    '''
    batch_norm_params={
        #decay for the moving average
        'decay':batch_norm_decay,
        #epsilon to prevent 0s in variance/防止除0错误
        'epsilon':batch_norm_epsilon,
        #collection cantaining update_ops
        'updates_collections':batch_norm_updatas_collections,
        #use fused batch norm if possible/是否使用更快、融合的实现方法?
        'fused':None,
    }
    if use_batch_norm:
        normalizer_fn=slim.batch_norm
        normalizer_params=batch_norm_params
    else:
        normalizer_fn=None
        normalizer_params={}
    #set weight decay for weights in conv and fc
    with slim.arg_scope([slim.conv2d,slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.conv2d],
                             weights_initializer=slim.variance_scaling_initializer(),
                             activation_fn=activation_fn,
                             normalizer_fn=normalizer_fn,
                             normalizer_params=normalizer_params) as sc:
            return sc

'''definition for inception v1 classifier'''
trunc_normal=lambda stddev:tf.truncated_normal_initializer(0.0,stddev)#截断的正态分布

def inception_v1_base(inputs,final_endpoint='Mixed_5c',scope='InceptionV1'):
    '''定义Inception V1的结构
    Args
        inputs: a tensor of size [batch_size,height,width,channels]
        final_endpoint: specifies the endpoint to construct the network
        scope: variable scope
    Return
        A dictionary from components of the network to the corresponding activation
    Raises
        ValueError: if final_endpoint is not set to one of the predefined values
    '''
    end_points={}
    with tf.variable_scope(scope,'InceptionV1',[inputs]):
        with slim.arg_scope([slim.conv2d,slim.fully_connected],
                            weights_initializer=trunc_normal(0.01)):
            with slim.arg_scope([slim.conv2d,slim.max_pool2d],
                                stride=1,padding='SAME'):
                end_point='Conv2d_1a_7x7'
                net=slim.conv2d(inputs,64,[7,7],stride=2,scope=end_point)
                end_points[end_point]=net
                if final_endpoint == end_point:
                    return net,end_point
                end_point='MaxPool_2a_3x3'
                net=slim.max_pool2d(net,[3,3],stride=2,scope=end_point)
                end_points[end_point]=net
                if final_endpoint == end_point:
                    return net,end_point
                end_point='Conv2d_2b_1x1'
                net=slim.conv2d(net,64,[1,1,scope=end_point])
                end_points[end_point]=net
                if final_endpoint == end_point:
                    return net,end_point
                end_point = 'Conv2d_2c_3x3'
                net = slim.conv2d(net, 192, [3, 3], scope=end_point)
                end_points[end_point] = net
                if final_endpoint == end_point: 
                    return net, end_points
                end_point = 'MaxPool_3a_3x3'
                net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
                end_points[end_point] = net
                if final_endpoint == end_point: 
                    return net, end_points

                end_point='Mixed_3b'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0=slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1=slim.conv2d(net,96,[1,1],scope='Conv2d_0a_1x1')
                        branch_1=slim.conv2d(branch_1,128,[3,3],scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2=slim.conv2d(net,16,[1,1],scope='Conv2d_0a_1x1')
                        branch_2=slim.conv2d(net,32,[5,5],scope='Conv2d_0b_5x5')
                    with tf.variable_scope('Branch_3'):
                        branch_3=slim.max_pool2d(net,[3,3],scope='MaxPool_0a_3x3')
                        branch_3=slim.conv2d(branch_3,32,[1,1],scope='Conv2d_0b_1x1')
                    net=tf.concat(axis=3,values=[branch_0,branch_1,branch_2,branch_3])
                end_points[end_point]=net
                if final_endpoint == end_point: 
                    return net, end_points

                end_point='Mixed_3c'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0=slim.conv2d(net,128,[1,1],scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1=slim.conv2d(net,128,[1,1],scope='Conv2d_0a_1x1')
                        branch_1=slim.conv2d(branch_1,192,[3,3],scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2=slim.conv2d(net,32,[1,1],scope='Conv2d_0a_1x1')
                        branch_2=slim.conv2d(net,96,[5,5],scope='Conv2d_0b_5x5')
                    with tf.variable_scope('Branch_3'):
                        branch_3=slim.max_pool2d(net,[3,3],scope='MaxPool_0a_3x3')
                        branch_3=slim.conv2d(branch_3,64,[1,1],scope='Conv2d_0b_1x1')
                    net=tf.concat(axis=3,values=[branch_0,branch_1,branch_2,branch_3])
                end_points[end_point]=net
                if final_endpoint == end_point: 
                    return net, end_points

                end_point='MaxPool_4a_3x3'
                net=slim.max_pool2d(net,[3,3],stride=2,scopr=end_point)
                end_points[end_point]=net
                if final_endpoint == end_point: 
                    return net, end_points

                end_point='Mixed_4b'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 208, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 48, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point: 
                    return net, end_points

                end_point = 'Mixed_4c'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 224, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(
                        axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point: 
                    return net, end_points

                end_point = 'Mixed_4d'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 256, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point: 
                    return net, end_points

                end_point = 'Mixed_4e'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 144, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 288, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point: 
                    return net, end_points

                end_point = 'Mixed_4f'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point: 
                    return net, end_points

                end_point = 'MaxPool_5a_2x2'
                net = slim.max_pool2d(net, [2, 2], stride=2, scope=end_point)
                end_points[end_point] = net
                if final_endpoint == end_point: 
                    return net, end_points

                end_point = 'Mixed_5b'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0a_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point: 
                    return net, end_points

                end_point = 'Mixed_5c'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 384, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point: 
                    return net, end_points
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

def inception_v1(inputs,
                 num_classes=1000,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='InceptionV1',
                 global_pool=False):
    #Final pooling and prediction
    with tf.variable_scope(scope,'InceptionV1',[inputs],reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm,slim.dropout],
                            is_training=is_training):
            net,end_points=inception_v1_base(inputs,scope=scope)
            with tf.variable_scope('Logits'):
                if global_pool:
                    net=tf.reduce_mean(net,[1,2],keep_dims=True,name='global_pool')
                    end_points['global_pool']=net
                else:
                    net=slim.avg_pool2d(net,[7,7],stride=1,scope='AvgPool_0a_7x7')
                    end_points['AvgPool_0a_7x7']=net
                if not num_classes:
                    return net,end_points
                net=slim.dropout(net,dropout_keep_prob,scope='Dropout_0b')
                logits=slim.conv2d(net,num_classes,[1,1],activation_fn=None,
                                   normalizer_fn=None,scope='Conv2d_0c_1x1')
                if spatial_squeeze:
                    logits=tf.squeeze(logits,[1,2],name='Spatial_squeeze')
                
                end_points['Logits'] = logits
                end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points

inception_v1.default_image_size = 224
inception_v1_arg_scope = inception_arg_scope


