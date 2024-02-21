import numpy as np
from pipeline_utils import get_visible_raw_image, get_metadata, normalize, white_balance, demosaic, \
    apply_color_space_transform, transform_xyz_to_srgb, apply_gamma, apply_tone_map, fix_orientation, \
    lens_shading_correction, Non_local_means_denoising
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import cv2
from model import RDUNet_qaunt, RDUNet
from inference import predict
import yaml
import torch.nn as nn
from PIL import Image
import glob
import os
import torch
import skimage.io
from inference import predict
import copy
import torch.ao.quantization as Quant
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder


def model_equivalence(model_1, model_2, device, rtol=1e-05, atol=1e-08, num_tests=100, input_size=(1,3,32,32)):

    model_1.to(device)
    model_2.to(device)

    for _ in range(num_tests):
        x = torch.rand(size=input_size).to(device)
        y1 = model_1(x).detach().cpu().numpy()
        y2 = model_2(x).detach().cpu().numpy()
        if np.allclose(a=y1, b=y2, rtol=rtol, atol=atol, equal_nan=False) == False:
            print("Model equivalence test sample failed: ")
            print(y1)
            print(y2)
            return False

    print("Model equivalence test pass")
    return True

def calibrate_model(model, loader, device=torch.device("cpu:0")):
    model.to(device)
    model.eval()

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        _ = model(inputs)

def prepare_loader(num_workers, train_batch_size):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset_path = '../capstone_design_m/python/Datasets'
    train_set = ImageFolder(root=train_dataset_path, transform=train_transform)

    # We will use test set for validation and test in this project.
    # Do not use test set for validation in practice!

    train_sampler = torch.utils.data.RandomSampler(train_set)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=train_batch_size,
        sampler=train_sampler, num_workers=num_workers)

    return train_loader

    

def run_pipeline(image_or_path, params=None, metadata=None, fix_orient=True):
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
    current_image = current_image[1100:1612, 1100:1612,...]
    
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


    # denoising part
        

    ######################### denoising with Non_local_means_denoising
    image_pipeline = Non_local_means_denoising(current_image)
    # image_pipeline: 기준이 되는 이미지! (정답이미지)
    
    ######################### rdu denoising 시작 #########################
    #model & test configuration load
    with open('../capstone_design_m/python/config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)

    model_params = config['model']
    test_params = config['test']
    n_channels = model_params['channels']
    
    #model parameters check
    print(model_params)
    
    #model generate
    model_path = os.path.join(test_params['pretrained models path'], 'real_small_model_div2k.pth')
    model = RDUNet(**model_params)
    
    device = torch.device("cpu")
    #model = model.to(device)
    #load model in single GPU
    #device = torch.device(test_params['device'])
    print("Using device: {}".format(device))

    #weight load
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device) #single GPU 사용 case
    model.eval()
    ######################### end of model generating #########################

    #save path setting
    if test_params['save images']:
        save_path = os.path.join(test_params['results path'])
    else:
        save_path = None
        
    #forward pass
    y_hat, image_w_model = predict(model, current_image, device, test_params['padding'],  n_channels, save_path)
    # current_image: RDUnet을 거친 우리가 만들어낸 모델의 이미지!
    ######################### end of rdu denoising #########################


    #gamma correction with rdunet
    print("start gamma correction")
    image_w_model = apply_gamma(image_w_model)

    #gamma correction without rdunet
    image_pipeline = apply_gamma(image_pipeline)   

    #tone mapping    
    print('Image Pipeline with RDUNet is done.')



    ####### PSNR, SSIM 값 계산 #######
    psnr = peak_signal_noise_ratio(image_pipeline, image_w_model, data_range=1.)
    ssim = structural_similarity(image_pipeline, image_w_model, data_range=1., multichannel=True,
                                 gaussian_weights=True, sigma=1.5, use_sample_covariance=False, win_size=3)

    print('Image: {} - PSNR: {:.4f} - SSIM: {:.4f}'.format(1, psnr, ssim))
    
    return image_w_model, image_pipeline