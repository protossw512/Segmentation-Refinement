CUDA_VISIBLE_DEVICES=1 python ../matting_test.py \
	--trimap_path=/media/wenxuan/LargeDisk/wangxiny/DAVIS/refinement_test/trimap/running \
	--rgb_path=/media/wenxuan/LargeDisk/wangxiny/DAVIS/refinement_test/rgb/running \
	--log_dir=/media/wenxuan/LargeDisk/wangxiny/seg_refine_train/DAVIS_1027_focal/test_log \
	--pred_path=/media/wenxuan/LargeDisk/wangxiny/DAVIS/refinement_test/pred/running \
	--save_ckpt_path=/media/wenxuan/LargeDisk/wangxiny/seg_refine_train/DAVIS_1027_focal/train \
	--dataset_name=DAVIS \
	--image_height=320 \
	--image_width=320 \
