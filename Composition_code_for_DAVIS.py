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
fg_path = '/media/wenxuan/LargeDisk/wangxiny/DAVIS/JPEGImages/480p/'

#path to provided alpha mattes
a_path = '/media/wenxuan/LargeDisk/wangxiny/DAVIS/Alphas/'

#Path to background images (MSCOCO)
bg_path = '/media/wenxuan/LargeDisk/wangxiny/Adobe_matting/train2017/'

tri_path = '/media/wenxuan/LargeDisk/wangxiny/DAVIS/SegPredictions/'

#Path to folder where you want the composited images to go
out_trimap_path = '/media/wenxuan/LargeDisk/wangxiny/DAVIS/trimap/'

out_path = '/media/wenxuan/LargeDisk/wangxiny/DAVIS/rgb/'

out_fg_path = '/media/wenxuan/LargeDisk/wangxiny/DAVIS/fg/'

out_bg_path = '/media/wenxuan/LargeDisk/wangxiny/DAVIS/bg/'

out_alpha_path = '/media/wenxuan/LargeDisk/wangxiny/DAVIS/alpha/'

##############################################################

from PIL import Image
import os 
import math
import time
import numpy as np
from scipy import misc
from joblib import Parallel, delayed


num_bgs = 5

tri_clips = os.listdir(tri_path)
bg_files = os.listdir(bg_path)

bg_iter = iter(bg_files)


def process_single(im_name):
    tri = Image.open(os.path.join(path_to_clip, tri_class, im_name))
    if os.path.exists(os.path.join(a_path, tri_clip, tri_class ,im_name)):
        a = Image.open(os.path.join(a_path, tri_clip, tri_class ,im_name))
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
        print (im.size, bg.size, ratio)
        if ratio > 1:        
            bg = bg.resize((math.ceil(bw*ratio),math.ceil(bh*ratio)), Image.BICUBIC)
        
        bg = bg.crop((0,0,w,h))
        
        fg = np.array(im)
        alpha = np.array(a)
        bng = np.array(bg)
        alpha_3 = np.dstack((alpha,alpha,alpha)) / 255.0
        out = fg * alpha_3 + bng * (1 - alpha_3)
        fg = fg * alpha_3
        
        #alpha = misc.imresize(alpha, (480,480), interp='bicubic')
        #bng = misc.imresize(bng, (480,480), interp='bicubic')
        #fg = misc.imresize(fg, (480, 480), interp='bicubic')
        #out = misc.imresize(out, (480,480), interp='bicubic')
        misc.imsave(out_trimap_path + tri_clip + tri_class + im_name[:len(im_name)-4] + '_' + str(bcount) + '.png', tri)
        misc.imsave(out_alpha_path + tri_clip + tri_class + im_name[:len(im_name)-4] + '_' + str(bcount) + '.png', alpha)
        misc.imsave(out_bg_path + tri_clip + tri_class + im_name[:len(im_name)-4] + '_' + str(bcount) + '.png', bng)
        misc.imsave(out_fg_path + tri_clip + tri_class + im_name[:len(im_name)-4] + '_' + str(bcount) + '.png', fg)
        misc.imsave(out_path + tri_clip + tri_class + im_name[:len(im_name)-4] + '_' + str(bcount) + '.png', out)
        bcount += 1
    return

for tri_clip in tri_clips:
    path_to_clip = os.path.join(tri_path, tri_clip)
    tri_classes = os.listdir(path_to_clip)
    for tri_class in tri_classes:
        full_tri_path = os.path.join(tri_path, tri_clip, tri_class)
        tri_files = os.listdir(full_tri_path)
        results = Parallel(n_jobs=4)(delayed(process_single)(img_name) for img_name in tri_files)
