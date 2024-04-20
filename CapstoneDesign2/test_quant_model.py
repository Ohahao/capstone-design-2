import yaml
import torch
import torch.optim as optim
import torch.quantization
import os
import copy
import sys
import numpy as np
import functools
from os.path import join
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from ptflops import get_model_complexity_info
import time

from model_quant import RDUNet
from model_qat6 import RDUNet_quant
from utils import set_seed  
    
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

    
def main():

    with open('config_qat.yaml', 'r') as stream:
        config_qat = yaml.safe_load(stream)

    model_params = config_qat['model']
    train_params = config_qat['train']

    cpu_device = torch.device("cpu:0")
    cuda_device = torch.device("cuda:4")
    model_dir = "saved_models"
    model_filename = "best_model.pth"
    quantized_model_filename = "best_model_quant_7bit.pth"
    model_filepath = os.path.join(model_dir, model_filename)
    quantized_model_filepath = os.path.join(model_dir, quantized_model_filename)


    #large model 로드
    model = RDUNet(channels=3, base_filters=64, device=cuda_device)
    model_fp32 = RDUNet(channels=3, base_filters=64, device=cpu_device)
    quantized_model = RDUNet_quant(model_fp32)
    model.load_state_dict(torch.load(model_filepath, map_location=cuda_device), strict=False)


    #quantized model 로드
    #양자화 설정
    quantization_config = torch.quantization.QConfig(
        activation=functools.partial(
            torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize,
            quant_min=0,
            quant_max=127
        ),
        weight=functools.partial(
            torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize,
            quant_min=-63,
            quant_max=64,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_affine
    )
    )

    quantized_model.qconfig = quantization_config

    quantized_model = torch.quantization.prepare_qat(quantized_model, inplace=True)
    quantized_model = torch.quantization.convert(quantized_model, inplace=True)   


    #양자화 모델 인스턴스화
    #quantized_model = RDUNet_quant(model_fp32)
    quantized_model.load_state_dict(torch.load(quantized_model_filepath))

    print("======== model load finished =========")

    #quantized_jit_model = load_torchscript_model(model_filepath=quantized_model_filepath, device=cpu_device)
    fp32_cpu_inference_latency = measure_inference_latency(model=model, device=cpu_device, input_size=(1,3,32,32), num_samples=100)
    int8_cpu_inference_latency = measure_inference_latency(model=quantized_model, device=cpu_device, input_size=(1,3,32,32), num_samples=100)
    #int8_jit_cpu_inference_latency = measure_inference_latency(model=quantized_jit_model, device=cpu_device, input_size=(1,3,32,32), num_samples=100)
    fp32_gpu_inference_latency = measure_inference_latency(model=model, device=cuda_device, input_size=(1,3,32,32), num_samples=100)
    
    print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(fp32_cpu_inference_latency * 1000))
    print("FP32 CUDA Inference Latency: {:.2f} ms / sample".format(fp32_gpu_inference_latency * 1000))
    print("INT7 CPU Inference Latency: {:.2f} ms / sample".format(int8_cpu_inference_latency * 1000))
    #print("INT8 JIT CPU Inference Latency: {:.2f} ms / sample".format(int8_jit_cpu_inference_latency * 1000))


    print("====== model size ========")

    '''
    print('Model summary:')
    test_shape = (model_params['channels'], train_params['patch size'], train_params['patch size'])
    with torch.no_grad():
        macs, params = get_model_complexity_info(model, test_shape, as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    
    print('\n\nQuantized Model summary:')
    test_shape = (model_params['channels'], train_params['patch size'], train_params['patch size'])
    with torch.no_grad():
        macs, params = get_model_complexity_info(quantized_model, test_shape, as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    '''
    model_size = os.path.getsize(model_filepath)  # 파일 크기를 바이트 단위로 측정
    model_size_MB = model_size / (1024 * 1024)
    quantized_model_size = os.path.getsize(quantized_model_filepath)
    quantized_model_size_MB = quantized_model_size / (1024 * 1024)

    print(f"Model file size: {model_size_MB:.2f} MB")
    print(f"QAT Model file size: {quantized_model_size_MB:.2f} MB")


if __name__ == '__main__':
    main()
    