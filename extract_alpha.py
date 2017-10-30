import numpy as np
import os
from PIL import Image
from scipy import misc

GTs_path = '/media/wenxuan/LargeDisk/wangxiny/DAVIS/Annotations/480p'
alphas_root = '/media/wenxuan/LargeDisk/wangxiny/DAVIS/Alphas'

def extract_alpha(GTs_path, alphas_root):
    class_folders = os.listdir(GTs_path)
    for class_folder in class_folders:
        annotations = os.listdir(os.path.join(GTs_path, class_folder))
        for annotation in annotations:
            filename = os.path.join(GTs_path, class_folder, annotation)
            img = Image.open(filename)
            img = np.array(img)
            indexes = np.unique(img)
            for index in range(1, max(indexes)+1):
                alpha_path = os.path.join(alphas_root, class_folder, '%d' % index)
                if not os.path.isdir(alpha_path):
                    os.makedirs(alpha_path)
                alpha_filename = os.path.join(alpha_path, annotation)
                alpha = np.copy(img)
                alpha[np.logical_and(alpha != 0, alpha != index)] = 0
                alpha[alpha > 0] = 255
                print alpha_filename
                misc.imsave(alpha_filename, alpha)


if __name__ == '__main__':
    extract_alpha(GTs_path, alphas_root)
