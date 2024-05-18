import glob
import os
import cv2
import time
import numpy as np
import torch
from pipeline import run_pipeline_v2
from pipeline_utils import get_visible_raw_image, get_metadata

params = {
    'save_as': 'png',
    'demosaic_type': 'EA',
    'save_dtype': np.uint8
}

# processing a directory 
print("==== Start ISP pipeline ====")
#image_paths = os.path.join(images_dir, 'short_exposure.dng')
#image_paths = '/home/hyoh/QAT_8/data/colorchart.dng'
image_paths = '/home/hyoh/Datasets/S7-ISP-Dataset/20161107_232916/medium_exposure.dng'
print('image path: ', image_paths)

#image_paths = 'S7-dataset 마지막 이미지'
#for image_path in image_paths:

# raw image data
raw_image = get_visible_raw_image(image_paths)

# metadata
metadata = get_metadata(image_paths)

# modify WB here
metadata['as_shot_neutral'] = [1., 1., 1.]


# 함수 실행 시간 측정 시작
start_time = time.time()

# render
print("==== Run ISP pipeline ====")
output_image, original_image = run_pipeline_v2(image_paths, params)

# 함수 실행 시간 측정 종료
end_time = time.time()
elapsed_time = end_time - start_time
print('ISP Pipeline Execution Time: {:.2f} seconds'.format(elapsed_time))

print('final output shape : {}'.format(output_image.shape))
print('original shape : {}'.format(original_image.shape))

# save // 저장 디렉토리: result
output_image_path1 = os.path.join('result', 'image_w_rdu_.' + params['save_as'])
output_image_path2 = os.path.join('result', 'image_wo_rdu_.' + params['save_as'])
max_val = 255
output_image = (output_image[..., ::-1] * max_val).astype(params['save_dtype'])
original_image = (original_image[..., ::-1] * max_val).astype(params['save_dtype'])
cv2.imwrite(output_image_path1, output_image)
cv2.imwrite(output_image_path2, original_image)
