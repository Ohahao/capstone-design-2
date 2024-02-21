import numpy as np
from pipeline_utils import get_visible_raw_image, get_metadata, normalize, white_balance, demosaic, \
    apply_color_space_transform, transform_xyz_to_srgb, apply_gamma, apply_tone_map, fix_orientation, \
    lens_shading_correction
import cv2
from model import RDUNet_qaunt
from model import RDUNet
from inference import predict
import yaml
import torch.nn as nn
from PIL import Image
import glob
import os
import torch
from PIL import Image
import skimage.io
from inference import predict
import copy
import torch.ao.quantization as Quant
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
#from metrics import PSNR, SSIM
import time
from ptflops import get_model_complexity_info
from torchstat import stat
from torchprofile import profile_macs


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

def prepare_loader(num_workers=8, train_batch_size=128):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset_path = 'Datasets'
    train_set = ImageFolder(root=train_dataset_path, transform=train_transform)

    # We will use test set for validation and test in this project.
    # Do not use test set for validation in practice!

    train_sampler = torch.utils.data.RandomSampler(train_set)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=train_batch_size,
        sampler=train_sampler, num_workers=num_workers)

    return train_loader

def save_torchscript_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)

def load_torchscript_model(model_filepath, device):

    model = torch.jit.load(model_filepath, map_location=device)

    return model

def measure_inference_latency(model,
                              device,
                              input_size=(1, 3, 32, 32),
                              num_samples=100,
                              num_warmups=10):

    model.to(device)
    model.eval()

    x = torch.rand(size=input_size).to(device)

    with torch.no_grad():
        for _ in range(num_warmups):
            _ = model(x)
    torch.cuda.synchronize()

    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_samples):
            _ = model(x)
            torch.cuda.synchronize()
        end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_ave = elapsed_time / num_samples

    return elapsed_time_ave    




def run_pipeline_v2(image_or_path, params=None, metadata=None, fix_orient=True):
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
    current_image = current_image[0:2500, 0:2500,...]
    
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

    
    

    ###rdu denoising 시작###
    cuda_device = torch.device("cuda:2")
    cpu_device = torch.device("cpu")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #model & test configuration load
    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)

    model_params = config['model']
    test_params = config['test']
    n_channels = model_params['channels']
    train_params = config['train']
 
    #model generate
    model_path = os.path.join(test_params['pretrained models path'], 'rdu_gaussian_32.pth')

    #weight load(pretrained model 불러오기)
    model_fp32 = RDUNet(**model_params)
    model = RDUNet(**model_params)      #for compare 
    state_dict = torch.load(model_path)
    model_fp32.load_state_dict(state_dict)
    model.load_state_dict(state_dict)

    model_fp32 = RDUNet_qaunt(model_fp32)
    model_before_fuse = copy.deepcopy(model_fp32)

    #cpu로 옮겨서 quantization
    model_fp32.to("cpu")

    model_fp32.eval()
    model.eval()

    #1. Customizing Configuration
    print("1. Customize Configuration")
    quantization_config = Quant.qconfig.QConfig(activation=Quant.MovingAverageMinMaxObserver.with_args(
                                                dtype=torch.quint8, qscheme=torch.per_tensor_symmetric), 
                                               weight=Quant.MovingAverageMinMaxObserver.with_args(
                                                   dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))
    model_fp32.qconfig = quantization_config
    backend = "fbgemm"
    torch.backends.quantized.engine = backend

    #2. fuse_module 적용
    #Fuse model
    print("2. fuse model")
    #model_for_fuse = torch.quantization.fuse_modules(model_for_fuse, [["conv1", "bn1", "relu"]], inplace=True)
    for module_name, module in model_fp32.named_children():
        if "DenoisingBlock" in module_name:
            for Denoising_block_name, Denoising_block in module.named_children():
                torch.quantization.fuse_modules(Denoising_block, [["conv_0", "actv_0"], ["conv_1", "actv_1"], ["conv_2", "actv_2"], ["conv_3", "actv_3"]], inplace=True)
        if "DownsampleBlock" in module_name:
            for Downsample_block_name, Downsample_block in Downsample_block.named_children():
                torch.quantization.fuse_modules(Downsample_block, [["conv", "actv"]], inplace=True)
        if "UpsampleBlock" in module_name:
            for Upsample_block_name, Upsample_block in Upsample_block.named_children():
                torch.quantization.fuse_modules(Upsample_block, [["conv_t", "actv_t"], ["conv", "actv"]], inplace=True)
        if "InputBlock" in module_name or "OutputBlock" in module_name :
            for inout_block_name, inout_block in inout_block.named_children():
                torch.quantization.fuse_modules(inout_block, [["conv_1", "actv_1"], ["conv_2", "actv_2"]], inplace=True)

    # Model and fused model should be equivalent.
    assert model_equivalence(model_1=model_fp32, model_2=model_before_fuse, device=cpu_device, rtol=1e-03, atol=1e-06, num_tests=100, input_size=(1, 3, 32, 32)), "Fused model is not equivalent to the original model!"
    
    #prepare model for PTSQ. Insert observers in model
    #qobserver = Quant.MinMaxObserver.with_args(dtype=torch.quint8)
    #wobserver = Quant.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    #quantized_model.qconfig = Quant.qconfig.QConfig(activation=qobserver, weight=wobserver)
    
    #3. prepare 수행
    print("3. preparate model")
    quantized_model_prepared = Quant.prepare(model_fp32)

    #4. Convert 수행
    print("4. convert model")
    train_loader = prepare_loader(num_workers=8, train_batch_size=128)
    calibrate_model(model=quantized_model_prepared, loader=train_loader, device=cpu_device) 
    model_int8 = Quant.convert(quantized_model_prepared, inplace=False)

    #quantized model 평가
    model_int8.eval()
    
    #모델 스펙 측정을 위해 모델 자체를 save
    torch.save(model_int8.state_dict(), 'Pretrained/model_int8_total.pth')
    #model load
    quantized_model = RDUNet(**model_params)
    state_dict = torch.load('Pretrained/model_int8_total.pth')
    quantized_model.load_state_dict(state_dict)

    #Save quantized model.
    save_torchscript_model(model=model_int8, model_dir='Pretrained', model_filename='model_int8.pt')
    # Load quantized model.
    quantized_jit_model = load_torchscript_model(model_filepath='Pretrained/model_int8.pt', device=cpu_device)

    print("finish quantize")
    
    ###end of model generating###

    #================ model 스펙 출력 - 기존 large model과 비교 ================#
    print('Large Model summary:')
    test_shape = (model_params['channels'], train_params['patch size'], train_params['patch size'])
    with torch.no_grad():
        macs, params = get_model_complexity_info(model, test_shape, as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        print("Model size: %.2f MB" %(os.path.getsize('Pretrained/rdu_gaussian_32.pth')/1e6))

    print('Quantizated Model summary:')
    test_shape = (model_params['channels'], train_params['patch size'], train_params['patch size'])
    with torch.no_grad():
        macs, params = get_model_complexity_info(quantized_model, test_shape, as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
        #stat(quantized_jit_model, test_shape)
        #macs = profile_macs(quantized_jit_model, test_shape)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        print("Model size: %.2f MB" %(os.path.getsize('Pretrained/model_int8_total.pth')/1e6))

    #=======================================================================#


    #===== model latency 측정 - 기존 large model과 비교 ======#
    fp32_cpu_inference_latency = measure_inference_latency(model=model, device=cpu_device, input_size=(1,3,32,32), num_samples=100)
    int8_cpu_inference_latency = measure_inference_latency(model=model_int8, device=cpu_device, input_size=(1,3,32,32), num_samples=100)
    int8_jit_cpu_inference_latency = measure_inference_latency(model=quantized_jit_model, device=cpu_device, input_size=(1,3,32,32), num_samples=100)
    fp32_gpu_inference_latency = measure_inference_latency(model=model, device=cuda_device, input_size=(1,3,32,32), num_samples=100)
    
    print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(fp32_cpu_inference_latency * 1000))
    print("FP32 CUDA Inference Latency: {:.2f} ms / sample".format(fp32_gpu_inference_latency * 1000))
    print("INT8 CPU Inference Latency: {:.2f} ms / sample".format(int8_cpu_inference_latency * 1000))
    print("INT8 JIT CPU Inference Latency: {:.2f} ms / sample".format(int8_jit_cpu_inference_latency * 1000))
    #=====================================================#

    #save path setting
    if test_params['save images']:
        save_path = os.path.join(test_params['results path'])
    else:
        save_path = None


    #forward pass
    y_hat, y_hat_ens = predict(model_int8, current_image, cpu_device, test_params['padding'],  n_channels, save_path)
    
    #기존의 large model (for compare)
    #y_hat_lg, y_hat_ens_lg = predict(model, current_image, device, test_params['padding'],  n_channels, save_path)


    ###end of rdu denoising###

    #gamma correction with rdunet
    print("start gamma correction")
    y_hat_ens = apply_gamma(y_hat_ens)

    #gamma correction without rdunet
    original_image = apply_gamma(current_image)   


    #tone mapping
    #y_hat_ens = apply_tone_map(y_hat_ens)
    
    print('Image Pipeline with RDUNet is done.')
    
    return y_hat_ens, original_image