import yaml
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
import torch.optim as optim
import os, sys
import copy
import numpy as np
import functools
from os.path import join
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from ptflops import get_model_complexity_info
import time

from model_quant import RDUNet
from model_quant import RDUNet_quant
from data_management import NoisyImagesDataset, DataSampler
from train_qat import fit_model
from transforms import AdditiveWhiteGaussianNoise, RandomHorizontalFlip, RandomVerticalFlip, RandomRot90
from utils import set_seed

def load_model(model, model_filepath, device):

    model.load_state_dict(torch.load(model_filepath, map_location=device), strict=False)

    return model

def load_torchscript_model(model_filepath, device):

    model = torch.jit.load(model_filepath, map_location=device)

    return model


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

    return True


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
    
    with open('config_qat.yaml', 'r') as stream:                # Load YAML configuration file.
        config_qat = yaml.safe_load(stream)

    model_params = config_qat['model']
    train_params = config_qat['train']
    val_params = config_qat['val']

    cuda_device = torch.device("cuda:1")
    cpu_device = torch.device("cpu:0")

    # Defining model:
    set_seed(0)
    model = RDUNet(channels=3, base_filters=64, device=cuda_device)

    '''
    print('Model summary:')
    test_shape = (model_params['channels'], train_params['patch size'], train_params['patch size'])
    with torch.no_grad():
        macs, params = get_model_complexity_info(model, test_shape, as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    '''

    #1. Define the model name and use multi-GPU if it is allowed.
    print("===== 1. Model 정의 및 로드하기 =====")
    
    model_dir = "saved_models"
    model_filename = "best_model.pth"
    quantized_model_filename = "best_model_quant_2.pth"
    model_filepath = os.path.join(model_dir, model_filename)
    quantized_model_filepath = os.path.join(model_dir, quantized_model_filename)
    
    model = load_model(model = model, model_filepath = model_filepath, device = cuda_device)
    
    #2. cpu에서 보낸 후 학습모드로 변경
    print("===== 2. Model 학습모드로 변경 =====")
    model.to(cpu_device)
    fused_model = copy.deepcopy(model) #layer fusion을 위한 model copy

    model.train()
    fused_model.train()
    
    #3. Fuse the model in place rather manually.
    print("===== 3. Fuse the model =====")
    for module_name, module in fused_model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                if basic_block_name == "DenoisingBlock":
                    torch.quantization.fuse_modules(basic_block, [["conv_0", "actv_0"], ["conv_1", "actv_1"], ["conv_2", "actv_2"], ["conv_3", "actv_3"]], inplace=True)
                elif basic_block_name == "InputBlock":
                    torch.quantization.fuse_modules(basic_block, [["conv_1", "actv_1"], ["conv_2", "actv_2"]], inplace=True)
                elif basic_block_name == "OutputBlock":
                    torch.quantization.fuse_modules(basic_block, [["conv_1", "actv_1"], ["conv_2", "actv_2"]], inplace=True)
                elif basic_block_name == "UpsampleBlock":
                    torch.quantization.fuse_modules(basic_block, [["conv", "actv"], ["conv_t", "actv_t"]], inplace=True)
                elif basic_block_name == "DownsampleBlock":
                    torch.quantization.fuse_modules(basic_block, [["conv", "actv"]], inplace=True)
                else:
                    print("error")
                    sys.exit()
                    

    #4. layer fusion 적용 확인
    # Model and fused model should be equivalent.
    print("===== 4. layer fusion 적용 확인 =====")
    model.eval()
    fused_model.eval()
    assert model_equivalence(model_1=model, model_2=fused_model, device=cpu_device, rtol=1e-03, atol=1e-06, num_tests=100, input_size=(1,3,32,32)), "Fused model is not equivalent to the original model!"

    #5. quantizaion model 불러오기
    print("===== 5. quantizaion model 불러오기 =====")
    quantized_model = RDUNet_quant(model_fp32=fused_model)
    
    #6. Quantization Configuration
    print("===== 6. Quantization Configuration =====")
    # quantizaion configuration 설정
    # quantization_config = torch.quantization.get_default_qat_qconfig("fbgemm") #나중에 고르기
    
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
   

    # Print quantization configurations
    print("quantiztion configurations")
    print(quantized_model.qconfig)
    
    print("===== 7. Quantization Preparation =====")
    torch.quantization.prepare_qat(quantized_model, inplace=True)

    # training data를 활용한 calibration 진행
    print("\nTraining QAT Model...")
    quantized_model.train()

    #======= train model ========#
    param_group = []
    for name, param in quantized_model.named_parameters():
        if 'conv' in name and 'weight' in name:
            p = {'params': param, 'weight_decay': train_params['weight decay']}
        else:
            p = {'params': param, 'weight_decay': 0.}
        param_group.append(p)

    # Load training and validation file names.
    # Modify .txt files if datasets do not fit in memory.
    files = os.listdir('/home/hyoh/Datasets/S7-ISP-Dataset')
    raw_train_gt_files = files[:5]    
    raw_train_n_files = files[:5]    
    raw_val_gt_files = files[5:8]    
    raw_val_n_files = files[5:8]    
    train_gt_files = list(map(lambda file: join(train_params['dataset path'], file, 'medium_exposure.jpg'), raw_train_gt_files))
    train_n_files = list(map(lambda file: join(train_params['dataset path'], file, 'short_exposure.jpg'), raw_train_n_files))
    val_gt_files = list(map(lambda file: join(train_params['dataset path'], file, 'medium_exposure.jpg'), raw_val_gt_files))        
    val_n_files = list(map(lambda file: join(train_params['dataset path'], file, 'short_exposure.jpg'), raw_val_n_files))

    training_transforms = transforms.Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRot90()
    ])

    # Predefined noise level
    train_noise_transform = [AdditiveWhiteGaussianNoise(train_params['noise level'], clip=True)]
    val_noise_transforms = [AdditiveWhiteGaussianNoise(s, fix_sigma=True, clip=True) for s in val_params['noise levels']]

    print('\nLoading training dataset:')
    training_dataset = NoisyImagesDataset(train_gt_files,
                                          train_n_files,
                                          model_params['channels'],
                                          train_params['patch size'],
                                          training_transforms)

    print('\nLoading validation dataset:')
    validation_dataset = NoisyImagesDataset(val_gt_files,
                                            val_n_files,
                                            model_params['channels'],
                                            val_params['patch size'],
                                            None)

    # Training in sub-epochs:
    print('Training patches:', len(training_dataset))
    print('Validation patches:', len(validation_dataset))
    n_samples = len(training_dataset) // train_params['dataset splits']
    n_epochs = train_params['epochs'] * train_params['dataset splits']
    sampler = DataSampler(training_dataset, num_samples=n_samples)

    data_loaders = {
        'train': DataLoader(training_dataset, train_params['batch size'], num_workers=train_params['workers'], sampler=sampler),
        'val': DataLoader(validation_dataset, val_params['batch size'], num_workers=val_params['workers']),
    }

    # Optimization:
    learning_rate = train_params['learning rate']
    step_size = train_params['scheduler step'] * train_params['dataset splits']

    criterion = nn.L1Loss()
    optimizer = optim.AdamW(param_group, lr=learning_rate)#설정 다시보기
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=train_params['scheduler gamma'])

    # Train the model / QAT가 적용된 FP model을 integer model로 변환 / quantized integer model 저장
    fit_model(quantized_model, data_loaders, model_params['channels'], criterion, optimizer, lr_scheduler, cuda_device,
              n_epochs, val_params['frequency'], train_params['checkpoint path'], quantized_model_filename)
    
    fp32_cpu_inference_latency = measure_inference_latency(model=model, device=cpu_device, input_size=(1,3,32,32), num_samples=100)
    fp32_gpu_inference_latency = measure_inference_latency(model=model, device=cuda_device, input_size=(1,3,32,32), num_samples=100)
    
    print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(fp32_cpu_inference_latency * 1000))
    print("FP32 CUDA Inference Latency: {:.2f} ms / sample".format(fp32_gpu_inference_latency * 1000))



if __name__ == '__main__':
    main()
    
