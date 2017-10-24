CUDA_VISIBLE_DEVICES=1 python ../matting_test.py \
	--trimap_path=/home/wangxiny/Segmentation-Refinement/GT04-tri.png \
	--rgb_path=/home/wangxiny/Segmentation-Refinement/GT04-rgb.png \
	--log_dir=/media/wenxuan/LargeDisk/wangxiny/seg_refine_train/adobe_1020_1/test_log \
	--save_ckpt_path=/media/wenxuan/LargeDisk/wangxiny/seg_refine_train/adobe_1020_1/train \
	--dataset_name=Adobe \
	--image_height=480 \
	--image_width=480 \
