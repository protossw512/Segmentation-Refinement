from __future__ import division
##Copyright 2017 Adobe Systems Inc.
##
##Licensed under the Apache License, Version 2.0 (the "License");
##you may not use this file except in compliance with the License.
##You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
##Unless required by applicable law or agreed to in writing, software
##distributed under the License is distributed on an "AS IS" BASIS,
##WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##See the License for the specific language governing permissions and
##limitations under the License.


##############################################################
#Set your paths here

#path to provided foreground images
fg_path = '/media/chenzeh/LargeDisk/wangxiny/DAVIS/JPEGImages/480p/'

#path to provided alpha mattes
a_path = '/media/chenzeh/LargeDisk/wangxiny/DAVIS/Alphas/'

#Path to background images (MSCOCO)
bg_path = '/media/chenzeh/LargeDisk/wangxiny/Adobe_matting/train2017/'

tri_path = '/media/chenzeh/LargeDisk/wangxiny/DAVIS/SegPredictions_clean'

ano_path = '/media/chenzeh/LargeDisk/wangxiny/DAVIS/Annotations/480p/'
#Path to folder where you want the composited images to go
out_trimap_path = '/media/chenzeh/LargeDisk/wangxiny/DAVIS/more_augmentation/trimap_clean/'

out_path = '/media/chenzeh/LargeDisk/wangxiny/DAVIS/more_augmentation/rgb_clean/'

out_fg_path = '/media/chenzeh/LargeDisk/wangxiny/DAVIS/more_augmentation/fg_clean/'

out_bg_path = '/media/chenzeh/LargeDisk/wangxiny/DAVIS/more_augmentation/bg_clean/'

out_alpha_path = '/media/chenzeh/LargeDisk/wangxiny/DAVIS/more_augmentation/alpha_clean/'

##############################################################

from PIL import Image
import os 
import math
import time
import numpy as np
from scipy import misc
from joblib import Parallel, delayed
import cv2
import pdb

num_bgs = 30

tri_clips = os.listdir(tri_path)
bg_files = os.listdir(bg_path)

bg_iter = iter(bg_files)


def process_single(im_name):
    tri = Image.open(os.path.join(path_to_clip, tri_class, im_name))
    if os.path.exists(os.path.join(a_path, tri_clip, tri_class ,im_name)):
        a = Image.open(os.path.join(a_path, tri_clip, tri_class ,im_name))
        ano = Image.open(os.path.join(ano_path,tri_clip, im_name))
    else:
        return
    im = Image.open(os.path.join(fg_path,tri_clip,im_name[:-3]+'jpg'))

    bbox = im.size
    w = bbox[0]
    h = bbox[1]
    
    if im.mode != 'RGB' and im.mode != 'RGBA':
        im = im.convert('RGB')
    
    bcount = 0 
    for i in range(num_bgs):

        bg_name = next(bg_iter)        
        print (im_name + '_' + str(bcount))
        #out.save(out_path + im_name[:len(im_name)-4] + '_' + str(bcount) + '.png', "PNG")
        if os.path.exists(out_alpha_path + tri_clip + tri_class + im_name[:len(im_name)-4] + '_' + str(bcount) + '.png') and \
                os.path.exists(out_trimap_path + tri_clip + tri_class + im_name[:len(im_name)-4] + '_' + str(bcount) + '.png') and \
                os.path.exists(out_fg_path + tri_clip + tri_class + im_name[:len(im_name)-4] + '_' + str(bcount) + '.png') and \
                os.path.exists(out_bg_path + tri_clip + tri_class + im_name[:len(im_name)-4] + '_' + str(bcount) + '.png') and \
                os.path.exists(out_path + tri_clip + tri_class + im_name[:len(im_name)-4] + '_' + str(bcount) + '.png'):
            bcount += 1
            continue
        bg = Image.open(bg_path + bg_name)
        if bg.mode != 'RGB':
            bg = bg.convert('RGB')
        bg_bbox = bg.size
        bw = bg_bbox[0]
        bh = bg_bbox[1]
        wratio = w / bw
        hratio = h / bh
        ratio = wratio if wratio > hratio else hratio     
        if ratio > 1:        
            #pdb.set_trace()
            bg = bg.resize((int(math.ceil(bw*ratio)),int(math.ceil(bh*ratio))), Image.BICUBIC)
        
        bg = bg.crop((0,0,w,h))
        
        fg = np.array(im)
        alpha = np.array(a)
        tri = np.array(tri)
        ano = np.array(ano)
        y, x = np.where(alpha > 0)
        if y != [] and x != []:
            center_idx = len(x) // 2
            center = (x[center_idx], y[center_idx])
        else:
            center = (h//2, w//2)
        angle = np.random.uniform(-30, 30)
        scale = np.random.uniform(0.5, 1.5)
        rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
        rot_mat = np.append(rot_mat, [[0,0,1]], axis=0)
        shift_x = np.random.randint(-50, 50)
        shift_y = np.random.randint(-75, 75)
        tran_mat = [[1,0,shift_x],[0,1,shift_y],[0,0,1]]
        M = np.matmul(rot_mat, tran_mat)
        fg_tr = cv2.warpAffine(fg, M[:2], fg.shape[1::-1], flags=cv2.INTER_NEAREST)
        alpha_tr = cv2.warpAffine(alpha, M[:2], fg.shape[1::-1], flags=cv2.INTER_NEAREST)
        tri_tr = cv2.warpAffine(tri, M[:2], fg.shape[1::-1], flags=cv2.INTER_NEAREST)
        ano_tr = cv2.warpAffine(ano, M[:2], fg.shape[1::-1], flags=cv2.INTER_NEAREST)
        #pdb.set_trace()
        if len(ano_tr.shape) == 2:
            ano_tr = np.expand_dims(ano_tr, axis=2)
        ano_grey = np.mean(ano_tr, -1)
        ano_mask = np.where(ano_grey>0, 1.0, 0.0)
        ano_3 = np.dstack((ano_mask, ano_mask, ano_mask))
        bng = np.array(bg)
        alpha_3 = np.dstack((alpha_tr,alpha_tr,alpha_tr)) / 255.0
        out = fg_tr * ano_3 + bng * (1.0 - ano_3)
        fg_tr = fg_tr * alpha_3
        #pdb.set_trace()
        #alpha = misc.imresize(alpha, (480,480), interp='bicubic')
        #bng = misc.imresize(bng, (480,480), interp='bicubic')
        #fg = misc.imresize(fg, (480, 480), interp='bicubic')
        #out = misc.imresize(out, (480,480), interp='bicubic')
        misc.imsave(out_trimap_path + tri_clip + tri_class + im_name[:len(im_name)-4] + '_' + str(bcount) + '.png', tri_tr)
        misc.imsave(out_alpha_path + tri_clip + tri_class + im_name[:len(im_name)-4] + '_' + str(bcount) + '.png', alpha_tr)
        misc.imsave(out_bg_path + tri_clip + tri_class + im_name[:len(im_name)-4] + '_' + str(bcount) + '.png', bng)
        misc.imsave(out_fg_path + tri_clip + tri_class + im_name[:len(im_name)-4] + '_' + str(bcount) + '.png', fg_tr)
        misc.imsave(out_path + tri_clip + tri_class + im_name[:len(im_name)-4] + '_' + str(bcount) + '.png', out)
        bcount += 1
    return

for tri_clip in tri_clips:
    path_to_clip = os.path.join(tri_path, tri_clip)
    tri_classes = os.listdir(path_to_clip)
    for tri_class in tri_classes:
        full_tri_path = os.path.join(tri_path, tri_clip, tri_class)
        tri_files = os.listdir(full_tri_path)
        results = Parallel(n_jobs=6)(delayed(process_single)(img_name) for img_name in tri_files)
