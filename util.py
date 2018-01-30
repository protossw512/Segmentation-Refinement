from joblib import Parallel, delayed
import tensorflow as tf
import numpy as np
import random
from scipy import misc,ndimage
import copy
import itertools
import os
from sys import getrefcount
import gc

trimap_kernel = [val for val in range(7,20)]
g_mean = np.array(([126.88,120.24,112.19])).reshape([1,1,3])
sample_patch_size = np.array([320, 480, 640, 800])
image_height = 320
image_width = 320


def image_preprocessing(image, is_training=False):
    if is_training:
        #distored_image, _ = distorted_bounding_box_crop(image)
        distored_image = tf.image.resize_images(image, [image_height, image_width])
        distored_image = tf.image.random_flip_left_right(distored_image)
        distored_image = tf.image.random_flip_up_down(distored_image)
        #distored_image = tf.image.random_saturation(distored_image, lower=0.5, upper=1.5)
        #distored_image = tf.image.random_brightness(distored_image, max_delta=32. / 255.)
        return distored_image
    else:
        return image

def distorted_bounding_box_crop(image,
                                bbox=tf.constant([0.0,0.0,1.0,1.0], dtype=tf.float32, shape=[1,1,4]),
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.5, 2.0),
                                area_range=(0.04, 0.64),
                                max_attempts=100,
                                scope=None):
  """Generates cropped_image using a one of the bboxes randomly distorted.
  See `tf.image.sample_distorted_bounding_box` for more documentation.
  Args:
    image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
      image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
      area of the image must contain at least this fraction of any bounding box
      supplied.
    aspect_ratio_range: An optional list of `floats`. The cropped area of the
      image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `floats`. The cropped area of the image
      must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
      region of the image of the specified constraints. After `max_attempts`
      failures, return the entire image.
    scope: Optional scope for name_scope.
  Returns:
    A tuple, a 3-D Tensor cropped_image and the distorted bbox
  """
  with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].

    # A large fraction of image datasets contain a human-annotated bounding
    # box delineating the region of the image containing the object of interest.
    # We choose to create a new bounding box for the object which is a randomly
    # distorted version of the human-annotated bounding box that obeys an
    # allowed range of aspect ratios, sizes and overlap with the human-annotated
    # bounding box. If no box is supplied, then we assume the bounding box is
    # the entire image.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    return cropped_image, distort_bbox

def unpool(pool, ind, ksize=[1, 2, 2, 1], scope='unpool'):
    
	with tf.variable_scope(scope):
		input_shape = pool.get_shape().as_list()
		output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

		flat_input_size = np.prod(input_shape)
		flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

		pool_ = tf.reshape(pool, [flat_input_size])
		batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
		b = tf.ones_like(ind) * batch_range
		b = tf.reshape(b, [flat_input_size, 1])
		ind_ = tf.reshape(ind, [flat_input_size, 1])
		ind_ = tf.concat([b, ind_], 1)

		ret = tf.scatter_nd(ind_, pool_, shape=flat_output_shape)
		ret = tf.reshape(ret, output_shape)
		return ret


def crop_patch(trimap,cropsize,dataset):
    border = cropsize / 2 - 1
    if dataset == "DAVIS":
        temp = np.where(trimap[border:-border-1, border:-border-1]>0)
    else:
        temp = np.where(trimap[border:-border-1, border:-border-1]==128)
    if len(temp[0])==0 or len(temp[1])==0:
	return None
    candidates = np.array([temp[0] + border, temp[1] + border])
    index = np.random.choice(len(candidates[0]))
    return [candidates[0][index], candidates[1][index]]

def load_path(alphas, trimaps, RGBs):
    '''
        rgb
            bike
                00001.jpg
                00002.jpg
                ...
            bear
            ...
        annotation
            bike
                00001.png
                00002.png
            bear
            ...
        segmentation
            bike
                1
                    00002.png
                    00003.png
                2
                    00002.png
                    00003.png
                ...
            ...

    '''
    class_folders = os.listdir(trimaps)
    alphas_abspath = []
    trimaps_abspath = []
    RGBs_abspath = []
    for class_folder in class_folders:
        object_folders = os.listdir(os.path.join(trimaps, class_folder))
        for object_folder in object_folders:
            masks = os.listdir(os.path.join(trimaps, class_folder, object_folder))
            for mask in masks:
                trimap = os.path.join(trimaps, class_folder, object_folder, mask)
                alpha = os.path.join(alphas, class_folder, object_folder, mask)
                RGB = os.path.join(RGBs, class_folder, mask[:-4] + '.jpg')
                if os.path.exists(alpha) and os.path.exists(RGB):
                    trimaps_abspath.append(trimap)
                    alphas_abspath.append(alpha)
                    RGBs_abspath.append(RGB)
    return np.array(alphas_abspath),np.array(trimaps_abspath),np.array(RGBs_abspath)

def load_data(batch_alpha_paths,batch_trimap_paths,batch_rgb_paths):
	print(batch_trimap_pathbatch_trimap_pathss)
	batch_size = batch_alpha_paths.shape[0]
	train_batch = []
	images_without_mean_reduction = []
	for i in range(batch_size):
			
		alpha = misc.imread(batch_alpha_paths[i],'L')
                if not (alpha.shape[0] == image_height and alpha.shape[1] == image_width):
                    alpha = misc.imresize(alpha, (image_height, image_width))
		trimap = misc.imread(batch_trimap_paths[i], 'P')
                if not (trimap.shape[0] == image_height and trimap.shape[1] == image_width):
                    trimap = misc.imresize(trimap, (image_height, image_width))
		rgb = misc.imread(batch_rgb_paths[i])
                if not (rgb.shape[0] == image_height and rgb.shape[1] == image_width):
                    rgb = misc.imresize(rgb, (image_height, image_width))
                
                alpha = np.expand_dims(alpha,2)
                trimap = np.expand_dims(trimap,2)
		
                fg = rgb.astype(np.float32) * np.concatenate([alpha.astype(np.float32), alpha.astype(np.float32), alpha.astype(np.float32)], axis=2) / 255.0
		batch_i = np.concatenate([alpha, trimap, rgb - g_mean, fg, rgb-fg, rgb],2)

                batch_i = batch_i.astype(np.float32)

		train_batch.append(batch_i)
	train_batch = np.stack(train_batch)
        #return np.expand_dims(train_batch[:,:,:,0],3),np.expand_dims(train_batch[:,:,:,1],3),train_batch[:,:,:,2:5], train_batch[:,:,:,5:8], train_batch[:,:,:,8:]
        return train_batch

def load_path_adobe(alphas,FGs, BGs, RGBs):
    '''
        rgb:
            0001.png
            ...
        fg:
            0001.png
            ...
        bg:
            0001.png
            ...
        alpha:
            0001.png
    '''
    image_names = os.listdir(alphas)
    alphas_abspath = []
    FGs_abspath = []
    BGs_abspath = []
    RGBs_abspath = []
    for image_name in image_names:
        alpha_path = os.path.join(alphas, image_name)
        FG_path = os.path.join(FGs, image_name)
        BG_path = os.path.join(BGs, image_name)
        RGB_path = os.path.join(RGBs, image_name)
        alphas_abspath.append(alpha_path)
        FGs_abspath.append(FG_path)
        BGs_abspath.append(BG_path)
        RGBs_abspath.append(RGB_path)
    return np.array(alphas_abspath),np.array(FGs_abspath),np.array(BGs_abspath),np.array(RGBs_abspath)

def load_single_image_adobe(alpha_path, FG_path, BG_path, RGB_path):
	alpha = misc.imread(alpha_path,'L')
	alpha = np.expand_dims(alpha,2)
	trimap = np.copy(alpha)
	trimap = generate_trimap(trimap, alpha)
	crop_size = np.random.choice(sample_patch_size)
	crop_center = crop_patch(trimap[:,:,0], crop_size, 'adobe')

	rgb = misc.imread(RGB_path)
	
	fg = misc.imread(FG_path)

	bg = misc.imread(BG_path)

	if crop_center is not None:
	    row_start = crop_center[0] - crop_size / 2 + 1
	    row_end = crop_center[0] + crop_size / 2 - 1
	    col_start = crop_center[1] - crop_size / 2 + 1
	    col_end = crop_center[1] + crop_size / 2 - 1
	    alpha = alpha[row_start:row_end, col_start:col_end, :]
	    rgb = rgb[row_start:row_end, col_start:col_end, :]
	    fg = fg[row_start:row_end, col_start:col_end, :]
	    bg = bg[row_start:row_end, col_start:col_end, :]
	if alpha.shape[0] != image_height:
	    alpha = np.expand_dims(misc.imresize(np.squeeze(alpha), (image_height,image_width)),2)
	    trimap = np.copy(alpha)
	    trimap = generate_trimap(trimap, alpha)
	    rgb = misc.imresize(rgb, (image_height,image_width))
	    fg = misc.imresize(fg, (image_height,image_width))
	    bg = misc.imresize(bg, (image_height,image_width))
	else:
	    trimap = np.copy(alpha)
	    trimap = generate_trimap(trimap, alpha)
	batch_i = np.concatenate([alpha, trimap, rgb - g_mean, fg, bg, rgb],2)
	batch_i = batch_i.astype(np.float32)
	return batch_i


def load_data_adobe(batch_alpha_paths,
                    batch_FG_paths,
                    batch_BG_paths,
                    batch_RGB_paths):
	
	batch_size = batch_alpha_paths.shape[0]
	train_batch = Parallel(n_jobs=8)(delayed(load_single_image_adobe)(batch_alpha_paths[i], \
				batch_FG_paths[i], batch_BG_paths[i], batch_RGB_paths[i]) \
				for i in range(batch_size))
	train_batch = np.stack(train_batch)
        #return np.expand_dims(train_batch[:,:,:,0],3),np.expand_dims(train_batch[:,:,:,1],3),train_batch[:,:,:,2:5], train_batch[:,:,:,5:8], train_batch[:,:,:,8:]
        return train_batch

def load_path_DAVIS(alphas,trimaps,FGs, BGs, RGBs):
    '''
        rgb:
            0001.png
            ...
        fg:
            0001.png
            ...
        bg:
            0001.png
            ...
        alpha:
            0001.png
    '''
    image_names = os.listdir(alphas)
    alphas_abspath = []
    trimaps_abspath = []
    FGs_abspath = []
    BGs_abspath = []
    RGBs_abspath = []
    for image_name in image_names:
        alpha_path = os.path.join(alphas, image_name)
        trimap_path = os.path.join(trimaps, image_name)
        FG_path = os.path.join(FGs, image_name)
        BG_path = os.path.join(BGs, image_name)
        RGB_path = os.path.join(RGBs, image_name)
        alphas_abspath.append(alpha_path)
        trimaps_abspath.append(trimap_path)
        FGs_abspath.append(FG_path)
        BGs_abspath.append(BG_path)
        RGBs_abspath.append(RGB_path)
    return np.array(alphas_abspath),np.array(trimaps_abspath),np.array(FGs_abspath),np.array(BGs_abspath),np.array(RGBs_abspath)

def load_single_image_DAVIS(alpha_path, trimap_path, FG_path, BG_path, RGB_path):
	alpha = misc.imread(alpha_path,'L')
	alpha = np.expand_dims(alpha,axis=2)
	trimap = misc.imread(trimap_path, 'L')
        trimap = np.expand_dims(trimap, axis=2)
	#trimap = generate_trimap(trimap, alpha)
	crop_size = np.random.choice(sample_patch_size)
	crop_center = crop_patch(trimap[:,:,0], crop_size, 'DAVIS')

	rgb = misc.imread(RGB_path)
	
	fg = misc.imread(FG_path)

	bg = misc.imread(BG_path)

	if crop_center is not None:
	    row_start = crop_center[0] - crop_size / 2 + 1
	    row_end = crop_center[0] + crop_size / 2 - 1
	    col_start = crop_center[1] - crop_size / 2 + 1
	    col_end = crop_center[1] + crop_size / 2 - 1
	    alpha = alpha[row_start:row_end, col_start:col_end, :]
            trimap = trimap[row_start:row_end, col_start:col_end, :]
            rgb = rgb[row_start:row_end, col_start:col_end, :]
	    fg = fg[row_start:row_end, col_start:col_end, :]
	    bg = bg[row_start:row_end, col_start:col_end, :]
	if alpha.shape[0] != image_height:
	    alpha = np.expand_dims(misc.imresize(np.squeeze(alpha), (image_height,image_width), interp='nearest'),2)
	    trimap = np.expand_dims(misc.imresize(np.squeeze(trimap), (image_height,image_width), interp='bicubic'),2)
	    rgb = misc.imresize(rgb, (image_height,image_width), interp='bicubic')
	    fg = misc.imresize(fg, (image_height,image_width))
	    bg = misc.imresize(bg, (image_height,image_width))
	batch_i = np.concatenate([alpha, trimap, rgb - g_mean, fg, bg, rgb],2)
	batch_i = batch_i.astype(np.float32)
	return batch_i


def load_data_DAVIS(batch_alpha_paths,
                    batch_trimap_paths,
                    batch_FG_paths,
                    batch_BG_paths,
                    batch_RGB_paths):
	
        batch_size = batch_alpha_paths.shape[0]
	train_batch = Parallel(n_jobs=8)(delayed(load_single_image_DAVIS)(batch_alpha_paths[i], \
				batch_trimap_paths[i], batch_FG_paths[i], batch_BG_paths[i], batch_RGB_paths[i]) \
				for i in range(batch_size))
	train_batch = np.stack(train_batch)
        #return np.expand_dims(train_batch[:,:,:,0],3),np.expand_dims(train_batch[:,:,:,1],3),train_batch[:,:,:,2:5], train_batch[:,:,:,5:8], train_batch[:,:,:,8:]
        return train_batch

def generate_trimap(trimap,alpha):

	k_size = random.choice(trimap_kernel)
        dilate = ndimage.grey_dilation(alpha[:,:,0],size=(k_size,k_size))
        erode = ndimage.grey_erosion(alpha[:,:,0],size=(k_size,k_size))
	# trimap[np.where((ndimage.grey_dilation(alpha[:,:,0],size=(k_size,k_size)) - ndimage.grey_erosion(alpha[:,:,0],size=(k_size,k_size)))!=0)] = 128
	trimap[np.where(dilate - erode>10)] = 128
	return trimap


def load_alphamatting_data(test_alpha):
	rgb_path = os.path.join(test_alpha,'rgb')
	trimap_path = os.path.join(test_alpha,'trimap')
	alpha_path = os.path.join(test_alpha,'alpha')	
	images = os.listdir(trimap_path)
	test_num = len(images)
	all_shape = []
	rgb_batch = []
	tri_batch = []
	alp_batch = []
	for i in range(test_num):
		rgb = misc.imread(os.path.join(rgb_path,images[i]))
		trimap = misc.imread(os.path.join(trimap_path,images[i]),'L')
		alpha = misc.imread(os.path.join(alpha_path,images[i]),'L')/255.0
		all_shape.append(trimap.shape)
		rgb_batch.append(misc.imresize(rgb,[320,320,3])-g_mean)
		trimap = misc.imresize(trimap,[320,320],interp = 'nearest').astype(np.float32)
		tri_batch.append(np.expand_dims(trimap,2))
		alp_batch.append(alpha)
	return np.array(rgb_batch),np.array(tri_batch),np.array(alp_batch),all_shape,images

def load_validation_data(vali_root):
	alpha_dir = os.path.join(vali_root,'alpha')
	RGB_dir = os.path.join(vali_root,'RGB')
	images = os.listdir(alpha_dir)
	test_num = len(images)
	
	all_shape = []
	rgb_batch = []
	tri_batch = []
	alp_batch = []

	for i in range(test_num):
		rgb = misc.imread(os.path.join(RGB_dir,images[i]))
		alpha = misc.imread(os.path.join(alpha_dir,images[i]),'L') 
		trimap = generate_trimap(np.expand_dims(np.copy(alpha),2),np.expand_dims(alpha,2))[:,:,0]
		alpha = alpha / 255.0
		all_shape.append(trimap.shape)
		rgb_batch.append(misc.imresize(rgb,[320,320,3])-g_mean)
		trimap = misc.imresize(trimap,[320,320],interp = 'nearest').astype(np.float32)
		tri_batch.append(np.expand_dims(trimap,2))
		alp_batch.append(alpha)
	return np.array(rgb_batch),np.array(tri_batch),np.array(alp_batch),all_shape,images
