import tensorflow as tf
import gpumemory
import numpy as np
from matting import image_preprocessing,load_path,load_data,load_path_adobe,load_data_adobe,load_alphamatting_data,load_validation_data,unpool
import os
from scipy import misc
import timeit

flags = tf.app.flags
flags.DEFINE_string('alpha_path', None, 'Path to alpha files')
flags.DEFINE_string('trimap_path', None, 'Path to trimap files')
flags.DEFINE_string('fg_path', None, 'Path to fg files')
flags.DEFINE_string('bg_path', None, 'Path to bg files')
flags.DEFINE_string('rgb_path', None, 'Path to rgb files')
flags.DEFINE_string('model_path', None, 'path to VGG weights')
flags.DEFINE_string('log_dir', None, 'Path to save logs')
flags.DEFINE_string('save_ckpt_path', None, 'Path to save ckpt files')
flags.DEFINE_string('save_meta_path', None, 'Path to save meta data')
flags.DEFINE_string('dataset_name', None, 'dataset name, "Adobe", "DAVIS"')
flags.DEFINE_integer('image_height', 320, 'input image height')
flags.DEFINE_integer('image_width', 320, 'input image width')
flags.DEFINE_integer('max_epochs', 500, 'max epochs to run' )
flags.DEFINE_integer('batch_size', 1, 'batch_size for training')
flags.DEFINE_integer('save_log_steps', 50, 'save log after steps')
flags.DEFINE_integer('save_ckpt_steps', 5000, 'save ckpt after steps')
flags.DEFINE_float('learning_rate', 0.0004, 'initial learning rate')
flags.DEFINE_float('learning_rate_decay', 0.95, 'learning rate decay factor')
flags.DEFINE_float('learning_rate_decay_steps', 100, 'learning rate decay after epochs')
flags.DEFINE_boolean('restore_from_ckpt', 'False', 'Whether restore weights form ckpt file')
flags.DEFINE_boolean('use_focal_loss', 'False', 'Whether use focal loss')

FLAGS = flags.FLAGS

def main(_):
    image_height = FLAGS.image_height
    image_width = FLAGS.image_width
    train_batch_size = FLAGS.batch_size
    max_epochs = FLAGS.max_epochs
    hard_mode = False

    #checkpoint file path
    #pretrained_model = './model/model.ckpt'
    pretrained_model = FLAGS.restore_from_ckpt
    #test_dir = './alhpamatting'
    #test_outdir = './test_predict'
    #validation_dir = '/data/gezheng/data-matting/new2/validation'

    #pretrained_vgg_model_path
    model_path = FLAGS.model_path
    log_dir = FLAGS.log_dir

    dataset_alpha = FLAGS.alpha_path
    dataset_trimap = FLAGS.trimap_path
    dataset_RGB = FLAGS.rgb_path
    dataset_fg = FLAGS.fg_path
    dataset_bg = FLAGS.bg_path
    if FLAGS.dataset_name == 'DAVIS':
        paths_alpha,paths_trimap,paths_RGB = load_path(dataset_alpha,dataset_trimap,dataset_RGB)
    else:
        paths_alpha, paths_FG, paths_BG, paths_RGB = load_path_adobe(dataset_alpha, dataset_fg, dataset_bg, dataset_RGB)

    range_size = len(paths_alpha)
    print('range_size is %d' % range_size)
    #range_size/batch_size has to be int
    batchs_per_epoch = int(range_size/train_batch_size) 

    index_queue = tf.train.range_input_producer(range_size, num_epochs=None,shuffle=True, seed=None, capacity=32)
    index_dequeue_op = index_queue.dequeue_many(train_batch_size, 'index_dequeue')

    train_batch = tf.placeholder(tf.float32, shape=(train_batch_size, image_height, image_width, 14))

    tf.add_to_collection('train_batch', train_batch)

    images = tf.map_fn(lambda img: image_preprocessing(img, is_training=True), train_batch)

    en_parameters = []
    pool_parameters = []
    
    b_GTmatte, b_trimap, b_RGB, b_GTFG, b_GTBG, raw_RGBs = tf.split(images, [1, 1, 3, 3, 3, 3], 3)

    tf.summary.image('GT_matte_batch',b_GTmatte,max_outputs = 4)
    tf.summary.image('trimap',b_trimap,max_outputs = 4)
    tf.summary.image('raw_RGBs',raw_RGBs,max_outputs = 4)

    b_input = tf.concat([b_RGB,b_trimap],3)

    # conv1_1
    with tf.name_scope('conv1_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 4, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(b_input, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1_1 = tf.nn.relu(out, name=scope)
        en_parameters += [kernel, biases]

    # conv1_2
    with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv1_2 = tf.nn.relu(out, name=scope)
            en_parameters += [kernel, biases]

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
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv2_1 = tf.nn.relu(out, name=scope)
        en_parameters += [kernel, biases]

    # conv2_2
    with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv2_2 = tf.nn.relu(out, name=scope)
            en_parameters += [kernel, biases]

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
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_1 = tf.nn.relu(out, name=scope)
        en_parameters += [kernel, biases]

    # conv3_2
    with tf.name_scope('conv3_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_2 = tf.nn.relu(out, name=scope)
        en_parameters += [kernel, biases]

    # conv3_3
    with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv3_3 = tf.nn.relu(out, name=scope)
            en_parameters += [kernel, biases]

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
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_1 = tf.nn.relu(out, name=scope)
        en_parameters += [kernel, biases]

    # conv4_2
    with tf.name_scope('conv4_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_2 = tf.nn.relu(out, name=scope)
        en_parameters += [kernel, biases]

    # conv4_3
    with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv4_3 = tf.nn.relu(out, name=scope)
            en_parameters += [kernel, biases]

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
                                             stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_1 = tf.nn.relu(out, name=scope)
        en_parameters += [kernel, biases]

    # conv5_2
    with tf.name_scope('conv5_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_2 = tf.nn.relu(out, name=scope)
        en_parameters += [kernel, biases]

    # conv5_3
    with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv5_3 = tf.nn.relu(out, name=scope)
            en_parameters += [kernel, biases]

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
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool5, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv6_1 = tf.nn.relu(out, name='conv6_1')

    training = True

    #deconv6
    with tf.variable_scope('deconv6') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 1, 512, 512], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv6_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        deconv6 = tf.nn.relu(tf.layers.batch_normalization(out,training=training), name='deconv6')
        
    deconv5_1 = unpool(deconv6,pool_parameters[-1])
        
    #deconv5_2
    with tf.variable_scope('deconv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([5, 5, 512, 512], dtype=tf.float32,stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(deconv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            deconv5_2 = tf.nn.relu(tf.layers.batch_normalization(out,training=training), name='deconv5_2')
            
            # V1 = tf.squeeze(tf.slice(deconv5_2,(0,0,0,0),(1,-1,-1,1)))
            # V2 = tf.squeeze(tf.slice(deconv5_2,(0,0,0,1),(1,-1,-1,1)))
            # V3 = tf.squeeze(tf.slice(deconv5_2,(0,0,0,2),(1,-1,-1,1)))
            # V4 = tf.squeeze(tf.slice(deconv5_2,(0,0,0,3),(1,-1,-1,1)))
            # V5 = tf.squeeze(tf.slice(deconv5_2,(0,0,0,4),(1,-1,-1,1)))
            # V = tf.expand_dims(tf.stack([V1,V2,V3,V4,V5]),3) 
            # tf.summary.image('deconv5_2',V,max_outputs = 5)

    deconv4_1 = unpool(deconv5_2,pool_parameters[-2])

    #deconv4_2
    with tf.variable_scope('deconv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([5, 5, 512, 256], dtype=tf.float32,stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(deconv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            deconv4_2 = tf.nn.relu(tf.layers.batch_normalization(out,training=training), name='deconv4_2')

            # V1 = tf.squeeze(tf.slice(deconv4_2,(0,0,0,0),(1,-1,-1,1)))
            # V2 = tf.squeeze(tf.slice(deconv4_2,(0,0,0,1),(1,-1,-1,1)))
            # V3 = tf.squeeze(tf.slice(deconv4_2,(0,0,0,2),(1,-1,-1,1)))
            # V4 = tf.squeeze(tf.slice(deconv4_2,(0,0,0,3),(1,-1,-1,1)))
            # V5 = tf.squeeze(tf.slice(deconv4_2,(0,0,0,4),(1,-1,-1,1)))
            # V = tf.expand_dims(tf.stack([V1,V2,V3,V4,V5]),3) 
            # tf.summary.image('deconv4_2',V,max_outputs = 5)

    deconv3_1 = unpool(deconv4_2,pool_parameters[-3])

    #deconv3_2
    with tf.variable_scope('deconv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([5, 5, 256, 128], dtype=tf.float32,stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(deconv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            deconv3_2 = tf.nn.relu(tf.layers.batch_normalization(out,training=training), name='deconv3_2')

            # V1 = tf.squeeze(tf.slice(deconv3_2,(0,0,0,0),(1,-1,-1,1)))
            # V2 = tf.squeeze(tf.slice(deconv3_2,(0,0,0,1),(1,-1,-1,1)))
            # V3 = tf.squeeze(tf.slice(deconv3_2,(0,0,0,2),(1,-1,-1,1)))
            # V4 = tf.squeeze(tf.slice(deconv3_2,(0,0,0,3),(1,-1,-1,1)))
            # V5 = tf.squeeze(tf.slice(deconv3_2,(0,0,0,4),(1,-1,-1,1)))
            # V = tf.expand_dims(tf.stack([V1,V2,V3,V4,V5]),3) 
            # tf.summary.image('deconv3_2',V,max_outputs = 5)

    deconv2_1 = unpool(deconv3_2,pool_parameters[-4])

    #deconv2_2
    with tf.variable_scope('deconv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([5, 5, 128, 64], dtype=tf.float32,stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(deconv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            deconv2_2 = tf.nn.relu(tf.layers.batch_normalization(out,training=training), name='deconv2_2')

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
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(deconv1_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        deconv1_2 = tf.nn.relu(tf.layers.batch_normalization(out,training=training), name='deconv1_2')
    #pred_alpha_matte
    with tf.variable_scope('pred_alpha') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 1], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(deconv1_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        pred_mattes = tf.nn.sigmoid(out)

    tf.add_to_collection("pred_mattes", pred_mattes)

    
    if FLAGS.dataset_name == 'DAVIS':
        wl = tf.ones_like(b_trimap)
    else:
	if FLAGS.use_focal_loss:
            wl = tf.where(tf.equal(b_trimap,128), tf.fill([train_batch_size,image_width,image_height,1],.5), tf.fill([train_batch_size,image_width,image_height,1], 0.))
        else:
            wl = tf.where(tf.equal(b_trimap,128), tf.fill([train_batch_size,image_width,image_height,1],1.), tf.fill([train_batch_size,image_width,image_height,1], 0.))
    tf.summary.image('pred_mattes',pred_mattes,max_outputs = 4)
    tf.summary.image('wl',wl,max_outputs = 4)
    #alpha_diff = tf.sqrt(tf.square(pred_mattes/255.0 - b_GTmatte/255.0,)  + 1e-12)
    if FLAGS.use_focal_loss:
   	alpha_diff = tf.square(pred_mattes - b_GTmatte/255.0,) + 1e-12
    else:
    	alpha_diff = tf.sqrt(tf.square(pred_mattes - b_GTmatte/255.0,) + 1e-12)

    p_RGB = []
    pred_mattes.set_shape([train_batch_size,image_height,image_width,1])
    b_GTBG.set_shape([train_batch_size,image_height,image_width,3])
    b_GTFG.set_shape([train_batch_size,image_height,image_width,3])
    raw_RGBs.set_shape([train_batch_size,image_height,image_width,3])
    b_GTmatte.set_shape([train_batch_size,image_height,image_width,1])

    # pred_final =  tf.where(tf.equal(b_trimap,128), pred_mattes, b_trimap/255.0)
    # tf.summary.image('pred_final',pred_final,max_outputs = 5)

    l_matte = tf.unstack(pred_mattes)
    BG = tf.unstack(b_GTBG)
    FG = tf.unstack(b_GTFG)

    for i in range(train_batch_size):
        #p_RGB.append(BG[i] - FG[i])
        #p_RGB.append((tf.ones_like(l_matte[i], dtype=tf.float32) - l_matte[i] / 255.0) * BG[i])
        #p_RGB.append(l_matte[i] / 255.0 * FG[i] + (tf.constant(1.) - l_matte[i] / 255.0) * BG[i])
        p_RGB.append(l_matte[i] * FG[i] +  (tf.constant(1.) - l_matte[i]) * BG[i])
        #p_RGB.append(l_matte[i] / 255.0 * FG[i] + (tf.constant(1.) - l_matte[i] / 255.0) * BG[i])
    pred_RGB = tf.stack(p_RGB)
    tf.summary.image('pred_RGB', pred_RGB, max_outputs = 4)
    tf.summary.image('GTFG', b_GTFG, max_outputs = 4)
    tf.summary.image('GTBG', b_GTBG, max_outputs = 4)
    #c_diff = tf.sqrt(tf.square(pred_RGB/255.0 - raw_RGBs/255.0) + 1e-12)
    # changed 201709
    # TODO figure out how to deal with this loss
    #c_diff = tf.sqrt(tf.square(pred_RGB/255.0 - raw_RGBs/255.0) + 1e-12)
    if FLAGS.use_focal_loss:
    	c_diff = tf.square(pred_RGB/255.0 - raw_RGBs/255.0) + 1e-12
    else:
    	c_diff = tf.sqrt(tf.square(pred_RGB/255.0 - raw_RGBs/255.0) + 1e-12)

    alpha_loss = tf.reduce_sum(alpha_diff) / tf.reduce_sum(wl) / 2.
    comp_loss = tf.reduce_sum(c_diff) / tf.reduce_sum(wl) / 2.
    #alpha_loss = tf.reduce_sum(alpha_diff * wl)/(tf.reduce_sum(wl))
    #comp_loss = tf.reduce_sum(c_diff * wl)/(tf.reduce_sum(wl))

    # tf.summary.image('alpha_diff',alpha_diff * wl_alpha,max_outputs = 5)
    # tf.summary.image('c_diff',c_diff * wl_RGB,max_outputs = 5)

    tf.summary.scalar('alpha_loss',alpha_loss)
    tf.summary.scalar('comp_loss',comp_loss)

    total_loss = (alpha_loss + comp_loss) * 0.5
    tf.summary.scalar('total_loss',total_loss)
    global_step = tf.Variable(0,name='global_step',trainable=False)

    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                                                global_step,
                                                FLAGS.learning_rate_decay_steps,
                                                FLAGS.learning_rate_decay,
                                                staircase=True,
                                                name='exponential_decay_learning_rate')
    tf.summary.scalar('learning_rate',learning_rate)
    train_op = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate).minimize(total_loss,global_step = global_step)

    #saver = tf.train.Saver(tf.trainable_variables() , max_to_keep = 10)
    saver = tf.train.Saver(max_to_keep = 10)

    coord = tf.train.Coordinator()
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.4)
    with tf.Session(config=tf.ConfigProto(gpu_options = gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(coord=coord,sess=sess)
        batch_num = 0
        epoch_num = 0
        #initialize all parameters in vgg16
        if not pretrained_model:
            weights = np.load(model_path)
            keys = sorted(weights.keys())
            for i, k in enumerate(keys):
                if i == 26:
                    break
                if k == 'conv1_1_W':  
                    sess.run(en_parameters[i].assign(np.concatenate([weights[k],np.zeros([3,3,1,64])],axis = 2)))
                else:
                    sess.run(en_parameters[i].assign(weights[k]))
            print('finish loading vgg16 model')
        else:
            print('Restoring pretrained model...')
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.save_ckpt_path))
            print('Restoring finished')
        sess.graph.finalize()
        epoch_num = global_step.eval() * train_batch_size // range_size
        while epoch_num < max_epochs:
            while batch_num < batchs_per_epoch:
                batch_index = sess.run(index_dequeue_op)
                total_start = timeit.default_timer()
                if FLAGS.dataset_name == 'DAVIS':
                    batch_alpha_paths = paths_alpha[batch_index]
                    batch_trimap_paths = paths_trimap[batch_index]
                    batch_RGB_paths = paths_RGB[batch_index]
                    images_batch = load_data(batch_alpha_paths,batch_trimap_paths,batch_RGB_paths)
                else:
                    batch_alpha_paths = paths_alpha[batch_index]
                    batch_FG_paths = paths_FG[batch_index]
                    batch_BG_paths = paths_BG[batch_index]
                    batch_RGB_paths = paths_RGB[batch_index]
                    images_batch = load_data_adobe(batch_alpha_paths,batch_FG_paths,batch_BG_paths,batch_RGB_paths)
                feed = {train_batch:images_batch}
                train_start = timeit.default_timer()
                _,loss,summary_str,step= sess.run([train_op,total_loss,summary_op,global_step],feed_dict = feed)
                train_end = timeit.default_timer()
                if step%FLAGS.save_ckpt_steps == 0:
                    saver.export_meta_graph(FLAGS.save_meta_path)
                    print('saving model......')
                    saver.save(sess,FLAGS.save_ckpt_path + '/model.ckpt',global_step = global_step, write_meta_graph = True)

                    print('test on validation data...')
                    #vali_diff = []
                    #test_RGBs,test_trimaps,test_alphas,all_shape,image_paths = load_validation_data(validation_dir)
                    #for i in range(len(test_RGBs)):
                    #    test_RGB = np.expand_dims(test_RGBs[i],0)
                    #    test_trimap = np.expand_dims(test_trimaps[i],0)
                    #    test_alpha = test_alphas[i]
                    #    shape_i = all_shape[i]
                    #    image_path = image_paths[i]
                    #    
                    #    feed = {image_batch:test_RGB,GT_trimap:test_trimap}
                    #    test_out = sess.run(pred_mattes,feed_dict = feed)
                    #    
                    #    i_out = misc.imresize(test_out[0,:,:,0],shape_i)
                    #    vali_diff.append(np.sum(np.abs(i_out/255.0-test_alpha))/(shape_i[0]*shape_i[1]))
                    #    misc.imsave(os.path.join(test_outdir,image_path),i_out)
                    #
                    #vali_loss = np.mean(vali_diff)
                    #print('validation loss is '+ str(vali_loss))
                    #validation_summary = tf.Summary()
                    #validation_summary.value.add(tag='validation_loss',simple_value = vali_loss)
                    #summary_writer.add_summary(validation_summary,step)
                if step%FLAGS.save_log_steps == 0:
                    summary_writer.add_summary(summary_str,global_step = step)
                batch_num += 1
                total_end = timeit.default_timer()
                print('epoch: %d, global_step: %d, loss is %f, batch_train_time: %f, batch_total_time: %f' \
                        %(epoch_num, step, loss, train_end - train_start, total_end - total_start))
            batch_num = 0
            epoch_num += 1

if __name__ == '__main__':
    tf.app.run()
