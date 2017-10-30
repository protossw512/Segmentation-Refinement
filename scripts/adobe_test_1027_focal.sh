CUDA_VISIBLE_DEVICES=1 python ../matting_test.py \
	--trimap_path=/home/wangxiny/Segmentation-Refinement/doll-tri.png \
	--rgb_path=/home/wangxiny/Segmentation-Refinement/doll-rgb.png \
	--log_dir=/media/wenxuan/LargeDisk/wangxiny/seg_refine_train/adobe_1027/test_log \
	--save_ckpt_path=/media/wenxuan/LargeDisk/wangxiny/seg_refine_train/adobe_1027/train \
	--dataset_name=Adobe \
	--image_height=320 \
	--image_width=320 \
