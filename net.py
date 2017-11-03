import tensorflow as tf
import numpy as np
from util import unpool

def refine_net(pred_mattes, b_RGB, trainable=True, training=True):
    pred_mattes_scaled = tf.scalar_mul(255.0, pred_mattes)
    b_input = tf.concat([b_RGB,pred_mattes_scaled],3)

    with tf.name_scope('ref_conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 4, 64], dtype=tf.float32,
                             stddev=1e-1), name='weights', trainable=trainable)
        conv = tf.nn.conv2d(b_input, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=trainable, name='biases')
        out = tf.nn.bias_add(conv, biases)
        out = tf.layers.batch_normalization(out, training=training, trainable=trainable)
        ref_conv1 = tf.nn.relu(out)
        tf.summary.histogram('weights', kernel)
        tf.summary.histogram('biases', biases)

    with tf.name_scope('ref_conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                             stddev=1e-1), name='weights', trainable=trainable)
        conv = tf.nn.conv2d(ref_conv1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=trainable, name='biases')
        out = tf.nn.bias_add(conv, biases)
        out = tf.layers.batch_normalization(out, training=training, trainable=trainable)
        ref_conv2 = tf.nn.relu(out)
        tf.summary.histogram('weights', kernel)
        tf.summary.histogram('biases', biases)

    with tf.name_scope('ref_conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                             stddev=1e-1), name='weights', trainable=trainable)
        conv = tf.nn.conv2d(ref_conv2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=trainable, name='biases')
        out = tf.nn.bias_add(conv, biases)
        out = tf.layers.batch_normalization(out, training=training, trainable=trainable)
        ref_conv3 = tf.nn.relu(out)
        tf.summary.histogram('weights', kernel)
        tf.summary.histogram('biases', biases)

    with tf.variable_scope('ref_pred_alpha') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 1], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=trainable)
        conv = tf.nn.conv2d(ref_conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                             trainable=trainable, name='biases')
        out = tf.nn.bias_add(conv, biases)
        ref_pred_mattes = tf.nn.sigmoid(out)
        tf.summary.histogram('weights', kernel)
        tf.summary.histogram('biases', biases)

    return ref_pred_mattes

def base_net(input_tensor, trainable=True, training=True):
    # conv1_1
    en_parameters = []
    pool_parameters = []
    with tf.name_scope('conv1_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 4, 64], dtype=tf.float32,
                             stddev=1e-1), name='weights', trainable=trainable)
        conv = tf.nn.conv2d(input_tensor, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=trainable, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1_1 = tf.nn.relu(out)
        en_parameters += [kernel, biases]
        tf.summary.histogram('weights', kernel)
        tf.summary.histogram('biases', biases)
    # conv1_2
    with tf.name_scope('conv1_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,stddev=1e-1), name='weights', trainable=trainable)
        conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),trainable=trainable, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1_2 = tf.nn.relu(out)
        en_parameters += [kernel, biases]
        tf.summary.histogram('weights', kernel)
        tf.summary.histogram('biases', biases)

        # V1 = tf.squeeze(tf.slice(conv1_2,(0,0,0,0),(1,-1,-1,1)))
        # V2 = tf.squeeze(tf.slice(conv1_2,(0,0,0,1),(1,-1,-1,1)))
        # V3 = tf.squeeze(tf.slice(conv1_2,(0,0,0,2),(1,-1,-1,1)))
        # V4 = tf.squeeze(tf.slice(conv1_2,(0,0,0,3),(1,-1,-1,1)))
        # V5 = tf.squeeze(tf.slice(conv1_2,(0,0,0,4),(1,-1,-1,1)))
        # V = tf.expand_dims(tf.stack([V1,V2,V3,V4,V5]),3)
        # tf.summary.image('conv1_2',V,max_outputs = 5)

    # pool1
    pool1,arg1 = tf.nn.max_pool_with_argmax(conv1_2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool1')
    pool_parameters.append(arg1)
    # conv2_1
    with tf.name_scope('conv2_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=trainable)
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                             trainable=trainable, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv2_1 = tf.nn.relu(out)
        en_parameters += [kernel, biases]
        tf.summary.histogram('weights', kernel)
        tf.summary.histogram('biases', biases)

    # conv2_2
    with tf.name_scope('conv2_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,stddev=1e-1), name='weights', trainable=trainable)
        conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=trainable, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv2_2 = tf.nn.relu(out)
        en_parameters += [kernel, biases]
        tf.summary.histogram('weights', kernel)
        tf.summary.histogram('biases', biases)
        # V1 = tf.squeeze(tf.slice(conv2_2,(0,0,0,0),(1,-1,-1,1)))
        # V2 = tf.squeeze(tf.slice(conv2_2,(0,0,0,1),(1,-1,-1,1)))
        # V3 = tf.squeeze(tf.slice(conv2_2,(0,0,0,2),(1,-1,-1,1)))
        # V4 = tf.squeeze(tf.slice(conv2_2,(0,0,0,3),(1,-1,-1,1)))
        # V5 = tf.squeeze(tf.slice(conv2_2,(0,0,0,4),(1,-1,-1,1)))
        # V = tf.expand_dims(tf.stack([V1,V2,V3,V4,V5]),3)
        # tf.summary.image('conv2_2',V,max_outputs = 5)

    # pool2
    pool2,arg2 = tf.nn.max_pool_with_argmax(conv2_2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool2')
    pool_parameters.append(arg2)
    # conv3_1
    with tf.name_scope('conv3_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=trainable)
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=trainable, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_1 = tf.nn.relu(out)
        en_parameters += [kernel, biases]
        tf.summary.histogram('weights', kernel)
        tf.summary.histogram('biases', biases)

    # conv3_2
    with tf.name_scope('conv3_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=trainable)
        conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=trainable, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_2 = tf.nn.relu(out)
        en_parameters += [kernel, biases]
        tf.summary.histogram('weights', kernel)
        tf.summary.histogram('biases', biases)

    # conv3_3
    with tf.name_scope('conv3_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,stddev=1e-1), name='weights', trainable=trainable)
        conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),trainable=trainable, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_3 = tf.nn.relu(out)
        en_parameters += [kernel, biases]
        tf.summary.histogram('weights', kernel)
        tf.summary.histogram('biases', biases)

        # V1 = tf.squeeze(tf.slice(conv3_3,(0,0,0,0),(1,-1,-1,1)))
        # V2 = tf.squeeze(tf.slice(conv3_3,(0,0,0,1),(1,-1,-1,1)))
        # V3 = tf.squeeze(tf.slice(conv3_3,(0,0,0,2),(1,-1,-1,1)))
        # V4 = tf.squeeze(tf.slice(conv3_3,(0,0,0,3),(1,-1,-1,1)))
        # V5 = tf.squeeze(tf.slice(conv3_3,(0,0,0,4),(1,-1,-1,1)))
        # V = tf.expand_dims(tf.stack([V1,V2,V3,V4,V5]),3)
        # tf.summary.image('conv3_3',V,max_outputs = 5)
    # pool3
    pool3,arg3 = tf.nn.max_pool_with_argmax(conv3_3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool3')
    pool_parameters.append(arg3)
    # conv4_1
    with tf.name_scope('conv4_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=trainable)
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=trainable, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_1 = tf.nn.relu(out)
        en_parameters += [kernel, biases]
        tf.summary.histogram('weights', kernel)
        tf.summary.histogram('biases', biases)

    # conv4_2
    with tf.name_scope('conv4_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=trainable)
        conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=trainable, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_2 = tf.nn.relu(out)
        en_parameters += [kernel, biases]
        tf.summary.histogram('weights', kernel)
        tf.summary.histogram('biases', biases)

    # conv4_3
    with tf.name_scope('conv4_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,stddev=1e-1), name='weights', trainable=trainable)
        conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),trainable=trainable, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_3 = tf.nn.relu(out)
        en_parameters += [kernel, biases]
        tf.summary.histogram('weights', kernel)
        tf.summary.histogram('biases', biases)

        # V1 = tf.squeeze(tf.slice(conv4_3,(0,0,0,0),(1,-1,-1,1)))
        # V2 = tf.squeeze(tf.slice(conv4_3,(0,0,0,1),(1,-1,-1,1)))
        # V3 = tf.squeeze(tf.slice(conv4_3,(0,0,0,2),(1,-1,-1,1)))
        # V4 = tf.squeeze(tf.slice(conv4_3,(0,0,0,3),(1,-1,-1,1)))
        # V5 = tf.squeeze(tf.slice(conv4_3,(0,0,0,4),(1,-1,-1,1)))
        # V = tf.expand_dims(tf.stack([V1,V2,V3,V4,V5]),3)
        # tf.summary.image('conv4_3',V,max_outputs = 5)

    # pool4
    pool4,arg4 = tf.nn.max_pool_with_argmax(conv4_3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool4')
    pool_parameters.append(arg4)
    # conv5_1
    with tf.name_scope('conv5_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights', trainable=trainable)
        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=trainable, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_1 = tf.nn.relu(out)
        en_parameters += [kernel, biases]
        tf.summary.histogram('weights', kernel)
        tf.summary.histogram('biases', biases)

    # conv5_2
    with tf.name_scope('conv5_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=trainable)
        conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=trainable, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_2 = tf.nn.relu(out)
        en_parameters += [kernel, biases]
        tf.summary.histogram('weights', kernel)
        tf.summary.histogram('biases', biases)

    # conv5_3
    with tf.name_scope('conv5_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,stddev=1e-1), name='weights', trainable=trainable)
        conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),trainable=trainable, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_3 = tf.nn.relu(out)
        en_parameters += [kernel, biases]

        tf.summary.histogram('weights', kernel)
        tf.summary.histogram('biases', biases)
        # V1 = tf.squeeze(tf.slice(conv5_3,(0,0,0,0),(1,-1,-1,1)))
        # V2 = tf.squeeze(tf.slice(conv5_3,(0,0,0,1),(1,-1,-1,1)))
        # V3 = tf.squeeze(tf.slice(conv5_3,(0,0,0,2),(1,-1,-1,1)))
        # V4 = tf.squeeze(tf.slice(conv5_3,(0,0,0,3),(1,-1,-1,1)))
        # V5 = tf.squeeze(tf.slice(conv5_3,(0,0,0,4),(1,-1,-1,1)))
        # V = tf.expand_dims(tf.stack([V1,V2,V3,V4,V5]),3)
        # tf.summary.image('conv5_3',V,max_outputs = 5)
    # pool5
    pool5,arg5 = tf.nn.max_pool_with_argmax(conv5_3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool4')
    pool_parameters.append(arg5)
    # conv6_1
    with tf.name_scope('conv6_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=trainable)
        conv = tf.nn.conv2d(pool5, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=trainable, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv6_1 = tf.nn.relu(out)
        tf.summary.histogram('weights', kernel)
        tf.summary.histogram('biases', biases)

    #deconv6
    with tf.variable_scope('deconv6') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 1, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=trainable)
        conv = tf.nn.conv2d(conv6_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=trainable, name='biases')
        out = tf.nn.bias_add(conv, biases)
        out = tf.layers.batch_normalization(out,training=training,trainable=trainable)
        deconv6 = tf.nn.relu(out)
        tf.summary.histogram('weights', kernel)
        tf.summary.histogram('biases', biases)

    deconv5_1 = unpool(deconv6,pool_parameters[-1])

    #deconv5_2
    with tf.variable_scope('deconv5_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 512, 512], dtype=tf.float32,stddev=1e-1), name='weights', trainable=trainable)
        conv = tf.nn.conv2d(deconv5_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),trainable=trainable, name='biases')
        out = tf.nn.bias_add(conv, biases)
        out = tf.layers.batch_normalization(out,training=training,trainable=trainable)
        deconv5_2 = tf.nn.relu(out)
        #TODO:change batch_normalization for rest deconvlayers
        # V1 = tf.squeeze(tf.slice(deconv5_2,(0,0,0,0),(1,-1,-1,1)))
        # V2 = tf.squeeze(tf.slice(deconv5_2,(0,0,0,1),(1,-1,-1,1)))
        # V3 = tf.squeeze(tf.slice(deconv5_2,(0,0,0,2),(1,-1,-1,1)))
        # V4 = tf.squeeze(tf.slice(deconv5_2,(0,0,0,3),(1,-1,-1,1)))
        # V5 = tf.squeeze(tf.slice(deconv5_2,(0,0,0,4),(1,-1,-1,1)))
        # V = tf.expand_dims(tf.stack([V1,V2,V3,V4,V5]),3)
        # tf.summary.image('deconv5_2',V,max_outputs = 5)
        tf.summary.histogram('weights', kernel)
        tf.summary.histogram('biases', biases)

    deconv4_1 = unpool(deconv5_2,pool_parameters[-2])

    #deconv4_2
    with tf.variable_scope('deconv4_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 512, 256], dtype=tf.float32,stddev=1e-1), name='weights', trainable=trainable)
        conv = tf.nn.conv2d(deconv4_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),trainable=trainable, name='biases')
        out = tf.nn.bias_add(conv, biases)
        out = tf.layers.batch_normalization(out,training=training,trainable=trainable)
        deconv4_2 = tf.nn.relu(out)

        # V1 = tf.squeeze(tf.slice(deconv4_2,(0,0,0,0),(1,-1,-1,1)))
        # V2 = tf.squeeze(tf.slice(deconv4_2,(0,0,0,1),(1,-1,-1,1)))
        # V3 = tf.squeeze(tf.slice(deconv4_2,(0,0,0,2),(1,-1,-1,1)))
        # V4 = tf.squeeze(tf.slice(deconv4_2,(0,0,0,3),(1,-1,-1,1)))
        # V5 = tf.squeeze(tf.slice(deconv4_2,(0,0,0,4),(1,-1,-1,1)))
        # V = tf.expand_dims(tf.stack([V1,V2,V3,V4,V5]),3)
        # tf.summary.image('deconv4_2',V,max_outputs = 5)
        tf.summary.histogram('weights', kernel)
        tf.summary.histogram('biases', biases)

    deconv3_1 = unpool(deconv4_2,pool_parameters[-3])

    #deconv3_2
    with tf.variable_scope('deconv3_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 256, 128], dtype=tf.float32,stddev=1e-1), name='weights', trainable=trainable)
        conv = tf.nn.conv2d(deconv3_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=trainable, name='biases')
        out = tf.nn.bias_add(conv, biases)
        out = tf.layers.batch_normalization(out,training=training,trainable=trainable)
        deconv3_2 = tf.nn.relu(out)

        # V1 = tf.squeeze(tf.slice(deconv3_2,(0,0,0,0),(1,-1,-1,1)))
        # V2 = tf.squeeze(tf.slice(deconv3_2,(0,0,0,1),(1,-1,-1,1)))
        # V3 = tf.squeeze(tf.slice(deconv3_2,(0,0,0,2),(1,-1,-1,1)))
        # V4 = tf.squeeze(tf.slice(deconv3_2,(0,0,0,3),(1,-1,-1,1)))
        # V5 = tf.squeeze(tf.slice(deconv3_2,(0,0,0,4),(1,-1,-1,1)))
        # V = tf.expand_dims(tf.stack([V1,V2,V3,V4,V5]),3)
        # tf.summary.image('deconv3_2',V,max_outputs = 5)
        tf.summary.histogram('weights', kernel)
        tf.summary.histogram('biases', biases)

    deconv2_1 = unpool(deconv3_2,pool_parameters[-4])

    #deconv2_2
    with tf.variable_scope('deconv2_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 128, 64], dtype=tf.float32,stddev=1e-1), name='weights', trainable=trainable)
        conv = tf.nn.conv2d(deconv2_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),trainable=trainable, name='biases')
        out = tf.nn.bias_add(conv, biases)
        out = tf.layers.batch_normalization(out,training=training,trainable=trainable)
        deconv2_2 = tf.nn.relu(out)

        tf.summary.histogram('weights', kernel)
        tf.summary.histogram('biases', biases)
        # V1 = tf.squeeze(tf.slice(deconv2_2,(0,0,0,0),(1,-1,-1,1)))
        # V2 = tf.squeeze(tf.slice(deconv2_2,(0,0,0,1),(1,-1,-1,1)))
        # V3 = tf.squeeze(tf.slice(deconv2_2,(0,0,0,2),(1,-1,-1,1)))
        # V4 = tf.squeeze(tf.slice(deconv2_2,(0,0,0,3),(1,-1,-1,1)))
        # V5 = tf.squeeze(tf.slice(deconv2_2,(0,0,0,4),(1,-1,-1,1)))
        # V = tf.expand_dims(tf.stack([V1,V2,V3,V4,V5]),3)
        # tf.summary.image('deconv2_2',V,max_outputs = 5)

    deconv1_1 = unpool(deconv2_2,pool_parameters[-5])

    #deconv1_2
    with tf.variable_scope('deconv1_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=trainable)
        conv = tf.nn.conv2d(deconv1_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=trainable, name='biases')
        out = tf.nn.bias_add(conv, biases)
        out = tf.layers.batch_normalization(out,training=training,trainable=trainable)
        deconv1_2 = tf.nn.relu(out)
        tf.summary.histogram('weights', kernel)
        tf.summary.histogram('biases', biases)
    #pred_alpha_matte
    with tf.variable_scope('pred_alpha') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 1], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=trainable)
        conv = tf.nn.conv2d(deconv1_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                             trainable=trainable, name='biases')
        out = tf.nn.bias_add(conv, biases)
        pred_mattes = tf.nn.sigmoid(out)
        tf.summary.histogram('weights', kernel)
        tf.summary.histogram('biases', biases)

    return pred_mattes, en_parameters
