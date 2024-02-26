import glob
import os
import cv2
import numpy as np

from pipeline import run_pipeline
from pipeline_utils import get_visible_raw_image, get_metadata

params = {
    'save_as': 'png',
    'demosaic_type': 'EA',
    'save_dtype': np.uint8
}

# processing a directory
images_dir = '../capstone_design_m/data/'
image_paths = glob.glob(os.path.join(images_dir, '*.dng'))
#image_paths = '../data/test.DNG'
for image_path in image_paths:
    # raw image data
    raw_image = get_visible_raw_image(image_path)

    # metadata
    metadata = get_metadata(image_path)

    # modify WB here
    metadata['as_shot_neutral'] = [1., 1., 1.]

    # render
    image_w_model, image_pipeline = run_pipeline(image_path, params)
    print('final output shape : {}'.format(image_w_model.shape))

    # save
    image_w_model_path = image_path.replace('.dng', '_{}.'.format('final_real_small_model_div2k') + params['save_as'])
    image_pipeline_path = image_path.replace('.dng', '_{}.'.format('final_pipeline') + params['save_as'])
    max_val = 255
    image_w_model = (image_w_model[..., ::-1] * max_val).astype(params['save_dtype'])
    image_pipeline = (image_pipeline[..., ::-1] * max_val).astype(params['save_dtype'])
    cv2.imwrite(image_w_model_path, image_w_model)
    cv2.imwrite(image_pipeline_path, image_pipeline)
