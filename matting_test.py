import tensorflow as tf
import gpumemory
import numpy as np
from matting import image_preprocessing,load_path,load_data,load_path_adobe,load_data_adobe,load_alphamatting_data,load_validation_data,unpool
import os
from scipy import misc
import timeit
from net import *

flags = tf.app.flags
flags.DEFINE_string('alpha_path', None, 'Path to alpha files')
flags.DEFINE_string('trimap_path', None, 'Path to trimap files')
flags.DEFINE_string('fg_path', None, 'Path to fg files')
flags.DEFINE_string('bg_path', None, 'Path to bg files')
flags.DEFINE_string('rgb_path', None, 'Path to rgb files')
flags.DEFINE_string('pred_path', None, 'Path to save alpha prediction files')
flags.DEFINE_string('model_path', None, 'path to VGG weights')
flags.DEFINE_string('log_dir', None, 'Path to save logs')
flags.DEFINE_string('save_ckpt_path', None, 'Path to save ckpt files')
flags.DEFINE_string('save_meta_path', None, 'Path to save meta data')
flags.DEFINE_string('dataset_name', None, 'dataset name, "Adobe", "DAVIS"')
flags.DEFINE_integer('image_height', 320, 'input image height')
flags.DEFINE_integer('image_width', 320, 'input image width')
flags.DEFINE_boolean('restore_from_ckpt', 'False', 'Whether restore weights form ckpt file')

g_mean = np.array(([126.88,120.24,112.19])).reshape([1,1,3])

FLAGS = flags.FLAGS

def main(_):
    image_height = FLAGS.image_height
    image_width = FLAGS.image_width

    #checkpoint file path
    #pretrained_model = './model/model.ckpt'
    pretrained_model = FLAGS.restore_from_ckpt
    
    log_dir = FLAGS.log_dir

    dataset_trimap = FLAGS.trimap_path
    dataset_RGB = FLAGS.rgb_path

    input_images = tf.placeholder(tf.float32, shape=(1, image_height, image_width, 4))

    tf.add_to_collection('input_images', input_images)

    b_trimap, b_RGB = tf.split(input_images, [1, 3], 3)

    tf.summary.image('trimap',b_trimap,max_outputs = 4)
    tf.summary.image('rgb',b_RGB,max_outputs = 4)

    b_input = tf.concat([b_RGB,b_trimap],3)

    with tf.name_scope('part1') as scope:
        pred_mattes, en_parameters = base_net(b_input, trainable=False, training=True)
    with tf.name_scope('part2') as scope:
        ref_pred_mattes = refine_net(pred_mattes, b_RGB, trainable=False, training=True)

    final_pred_mattes = tf.add(pred_mattes, ref_pred_mattes)
    
    tf.add_to_collection("pred_mattes", pred_mattes)

    
    tf.summary.image('pred_mattes',pred_mattes,max_outputs = 4)
    pred_mattes.set_shape([1,image_height,image_width,1])

    global_step = tf.Variable(0,name='global_step',trainable=False)

    coord = tf.train.Coordinator()
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
    saver = tf.train.Saver(max_to_keep = 10)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.4)
    with tf.Session(config=tf.ConfigProto(gpu_options = gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(coord=coord,sess=sess)
        print('Restoring pretrained model...')
        saver.restore(sess,tf.train.latest_checkpoint(FLAGS.save_ckpt_path))
        global_step.assign(0).eval()
        print('Restoring finished')
        sess.graph.finalize()
        if os.path.isdir(dataset_trimap):
            trimap_files = os.listdir(dataset_trimap)
        else:
            trimap_files = dataset_trimap
        for trimap_file in trimap_files:
            trimap_path = os.path.join(dataset_trimap, trimap_file)
            RGB_path = os.path.join(dataset_RGB, trimap_file[:-3]+'jpg')
            trimap_img = misc.imread(trimap_path, 'L')
            ori_shape = trimap_img.shape
            trimap_img = np.expand_dims(np.expand_dims(misc.imresize(trimap_img.astype(np.uint8), [image_height, image_width], interp='nearest').astype(np.float32),2),0)
            rgb_img = misc.imread(RGB_path)
            rgb_img = np.expand_dims(misc.imresize(rgb_img.astype(np.uint8), [image_height, image_width]).astype(np.float32) - g_mean, 0)
            image = np.concatenate([trimap_img, rgb_img], axis=3)
            feed = {input_images:image}
            train_start = timeit.default_timer()
            pred_alpha, summary_str, step = sess.run([pred_mattes,summary_op,global_step], feed_dict = feed)
            summary_writer.add_summary(summary_str,global_step=step)
            pred_alpha = np.squeeze(pred_alpha)
            pred_alpha = misc.imresize(pred_alpha, ori_shape)
            misc.imsave(os.path.join(FLAGS.pred_path, trimap_file), pred_alpha)
            train_end = timeit.default_timer()
            print ('global_step:%d, time:%f' % (step, train_end-train_start))

if __name__ == '__main__':
    tf.app.run()
