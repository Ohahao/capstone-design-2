import numpy as np
from pipeline_utils import get_visible_raw_image, get_metadata, normalize, white_balance, demosaic, \
    apply_color_space_transform, transform_xyz_to_srgb, apply_gamma, apply_tone_map, fix_orientation, \
    lens_shading_correction, Non_local_means_denoising
import cv2
import os, sys
#from model import RDUNet    #원본 모델 적용 시
sys.path.append("/home/hyoh/QAT")   #양자화 모델 적용 시
from model_qat8 import RDUNet_quant
from model_qat8 import RDUNet
from inference import predict
import yaml
import torch.nn as nn
from PIL import Image
import glob
import torch
import gc
import functools
from PIL import Image
import skimage.io




def run_pipeline_v2(image_or_path, params=None, metadata=None, fix_orient=True):
    #del unused_variable
    gc.collect()

    params_ = params.copy()
    if type(image_or_path) == str:
        image_path = image_or_path
        # raw image data
        raw_image = get_visible_raw_image(image_path)
        # metadata
        metadata = get_metadata(image_path)
    else:
        raw_image = image_or_path.copy()
        # must provide metadata
        if metadata is None:
            raise ValueError("Must provide metadata when providing image data in first argument.")

    current_image = raw_image
    #reshape input image
    current_image = current_image[1000:2000, 1000:2000,...]
    
    # linearization
    linearization_table = metadata['linearization_table']
    if linearization_table is not None:
        print('Linearization table found. Not handled.')

    #normalizing
    current_image = normalize(current_image, metadata['black_level'], metadata['white_level'])
    
    #lens shading correction
    gain_map_opcode = None
    if 'opcode_lists' in metadata:
        if 51009 in metadata['opcode_lists']:
            opcode_list_2 = metadata['opcode_lists'][51009]
            gain_map_opcode = opcode_list_2[9]
    if gain_map_opcode is not None:
        current_image = lens_shading_correction(current_image, gain_map_opcode=gain_map_opcode, bayer_pattern=metadata['cfa_pattern'])

    #white balancing
    current_image = white_balance(current_image, metadata['as_shot_neutral'], metadata['cfa_pattern'])

    #demosaicing
    current_image = demosaic(current_image, metadata['cfa_pattern'], output_channel_order='RGB', alg_type=params_['demosaic_type'])

    #color space transform
    current_image = apply_color_space_transform(current_image, metadata['color_matrix_1'], metadata['color_matrix_2'])

    #transform xyz to srgb
    current_image = transform_xyz_to_srgb(current_image)
    
    if fix_orient:
        # fix image orientation, if needed (after srgb stage, ok?)
        current_image = fix_orientation(current_image, metadata['orientation'])
   
    #################################
    ###rdu denoising 시작#############
    #model & test configuration load
    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)

    test_params = config['test']
    model_params = config['model']
    n_channels = model_params['channels']

    cpu_device = torch.device("cpu:0")
    cuda_device = torch.device("cuda:3")
    model_dir = "/home/hyoh/QAT/saved_models"
    model_filename = "best_model.pth"
    quantized_model_filename = "best_model_quant_8bit.pth"
    model_filepath = os.path.join(model_dir, model_filename)
    quantized_model_filepath = os.path.join(model_dir, quantized_model_filename)
    
    #model parameters check
    print(model_params)

    '''
    #======== 원본 모델 적용 ==========#
    model_fp32 = RDUNet(**model_params) 
    # 저장된 양자화 모델 상태 딕셔너리 로드
    state_dict = torch.load(model_filepath)
    # 양자화 모델에 상태 딕셔너리 적용
    model_fp32.load_state_dict(state_dict, strict=False)
    # 모델을 평가 모드로 설정 (필요한 경우)
    model_fp32.eval()
    
    
    
    #======== 양자화 모델 적용 ==========#
    # Sub-8bit Quantized 모델 load
    model_fp32 = RDUNet(last_calibrate=False, quant=False, calibrate=False, convert=False, device=cuda_device, **model_params) #model generate
    quantized_model = RDUNet_quant(model_fp32, device=cuda_device)
    # state_dict key의 차원 맞춰주기
    state_dict = torch.load(quantized_model_filepath)
    # .weight로 끝나는 키만 사용하여 모델의 가중치를 설정
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.endswith('.weight_quantized')}

    quantized_model.load_state_dict(filtered_state_dict, strict=False)
    
    '''
    #======== 8bit QAT model 적용 =========# 
    #large model 로드
    model = RDUNet(channels=3, base_filters=64, device=cpu_device)
    model_fp32 = RDUNet(channels=3, base_filters=64, device=cpu_device)
    quantized_model = RDUNet_quant(model_fp32)
    model.load_state_dict(torch.load(model_filepath, map_location=cuda_device), strict=False)

    
    #cpu 사용!!
    #양자화 설정
    quantization_config = torch.quantization.QConfig(
        activation=functools.partial(
            torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize,
            quant_min=0,
            quant_max=255
        ),
        weight=functools.partial(
            torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_affine
    )
    )

    quantized_model.qconfig = quantization_config
    quantized_model = torch.quantization.prepare(quantized_model, inplace=True)
    quantized_model = torch.quantization.convert(quantized_model, inplace=True)   
    
    #양자화 모델 인스턴스화
    #quantized_model = RDUNet_quant(model_fp32)
    quantized_model.load_state_dict(torch.load(quantized_model_filepath), strict=False)
    

    #inference model 선택
    model = quantized_model
    #model = model_fp32

    #load model in single GPU
    device = torch.device('cpu:0')
    #device = torch.device(cuda_device)
    print("Using device: {}".format(device))
    
    ###end of model generating###
    
    #save path setting
    if test_params['save images']:
        save_path = os.path.join(test_params['results path'])
    else:
        save_path = None

    #forward pass
    #y_hat : original image
    #y_hat_ens : ensamble result
    y_hat, y_hat_ens = predict(model, current_image, device, test_params['padding'],  n_channels, save_path)
    #########end of rdu denoising###########
    ########################################
    
    ########### Denoising without rdunet ###########
    current_image = Non_local_means_denoising(current_image)

    #########end of denoising###########
    ########################################


    #gamma correction with rdunet
    y_hat_ens = apply_gamma(y_hat_ens)
    #gamma correction without rdunet
    original_image = apply_gamma(current_image)   


    #tone mapping
    #y_hat_ens = apply_tone_map(y_hat_ens)
    
    print('Image Pipeline with RDUNet is done.')
    
    return y_hat_ens, original_image
