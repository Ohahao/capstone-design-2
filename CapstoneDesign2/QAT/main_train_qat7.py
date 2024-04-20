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
from tqdm import tqdm

from model_qat7 import RDUNet
from model_qat7 import RDUNet_quant
from data_management import NoisyImagesDataset, DataSampler
from train_qat7 import fit_model
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


def prepare_dataloader(model_params, train_params, val_params, num_train=90, num_val=10, num_calibrate=2):

    # Load training and validation file names.
    # Modify .txt files if datasets do not fit in memory.
    files = os.listdir('/home/hyoh/Datasets/S7-ISP-Dataset')
    raw_calibrate_files = files[:num_calibrate] 
    raw_train_gt_files = files[:num_train]    
    raw_train_n_files = files[:num_train]    
    raw_val_gt_files = files[num_train:num_train + num_val]    
    raw_val_n_files = files[num_train:num_train + num_val]    
    calibrate_gt_files = list(map(lambda file: join(train_params['dataset path'], file, 'medium_exposure.jpg'), raw_calibrate_files))
    calibrate_n_files = list(map(lambda file: join(train_params['dataset path'], file, 'short_exposure.jpg'), raw_calibrate_files))
    train_gt_files = list(map(lambda file: join(train_params['dataset path'], file, 'medium_exposure.jpg'), raw_train_gt_files))
    train_n_files = list(map(lambda file: join(train_params['dataset path'], file, 'short_exposure.jpg'), raw_train_n_files))
    val_gt_files = list(map(lambda file: join(train_params['dataset path'], file, 'medium_exposure.jpg'), raw_val_gt_files))        
    val_n_files = list(map(lambda file: join(train_params['dataset path'], file, 'short_exposure.jpg'), raw_val_n_files))

    training_transforms = transforms.Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRot90()
    ])

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
    
    print('\nLoading calibration dataset:')
    calibration_dataset = NoisyImagesDataset( calibrate_gt_files,
                                        calibrate_n_files,
                                        model_params['channels'],
                                        train_params['patch size'],
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
        'calibrate': DataLoader(calibration_dataset, train_params['batch size'], num_workers=0)
    }

    return data_loaders


def main():
    
    with open('config_qat.yaml', 'r') as stream:                # Load YAML configuration file.
        config_qat = yaml.safe_load(stream)

    model_params = config_qat['model']
    train_params = config_qat['train']
    val_params = config_qat['val']
    quant_params = config_qat['quant']


    cuda_device = torch.device("cuda:1")
    cpu_device = torch.device("cpu:0")

    # Defining model:
    set_seed(0)
    model = RDUNet(last_calibrate=False, **model_params)

    model_dir = "saved_models"
    model_filename = "best_model.pth"
    quantized_model_filename = "best_model_quant_7bit.pth"
    model_filepath = os.path.join(model_dir, model_filename)
    quantized_model_filepath = os.path.join(model_dir, quantized_model_filename)

    #======= 1. Pretrained model 로드 ========#
    print("===== 1. Pretrained model 로드 =====")
    model = load_model(model = model, model_filepath = model_filepath, device = cuda_device)

    #2. Fuse the model in place rather manually.
    print("===== 2. layer fusion =====")
    model.to(cpu_device)
    fused_model = copy.deepcopy(model) #layer fusion을 위한 model copy

    model.train()
    fused_model.train()

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
                    

    #3. layer fusion 적용 확인
    # Model and fused model should be equivalent.
    print("===== 3. layer fusion 적용 확인 =====")
    model.eval()
    fused_model.eval()
    assert model_equivalence(model_1=model, model_2=fused_model, device=cpu_device, rtol=1e-03, atol=1e-06, num_tests=100, input_size=(1,3,32,32)), "Fused model is not equivalent to the original model!"


    #4. Quantized model 불러오기
    print("===== 4. Quantized model 불러오기 =====")
    quantized_model = RDUNet_quant(model_fp32=fused_model)

    #5. Prepare QAT
    print("===== 5. Prepare QAT =====")
    quantized_model.to(cpu_device)
    quantized_model.eval()

    data_loaders = prepare_dataloader(model_params, train_params, val_params, num_train=5, num_val=3, num_calibrate=2)
    calibration_loader = data_loaders['calibrate']

    #calibrate / quant 설정 변경
    quantized_model.model_fp32.set_quant_calibrate(quant=False, calibrate=True)

    #calibrate 실행
    total_batches = len(calibration_loader)
    with torch.no_grad():
        for idx, (inputs, labels) in tqdm(enumerate(calibration_loader), total=total_batches, desc="Processing"):

            # 현재 배치가 마지막 배치인 경우
            if idx == total_batches - 1:
                model.last_calibrate = True
            
            quantized_model(inputs)

    # 6. QAT training 진행
    print("===== 6. QAT Training 진행 =====")
    print("\nTraining QAT Model...")
    quantized_model.train()

    #calibrate / quant 설정 변경
    quantized_model.model_fp32.set_quant_calibrate(quant=True, calibrate=False)

    #======= train model ========#
    param_group = []
    for name, param in quantized_model.named_parameters():
        if 'conv' in name and 'weight' in name:
            p = {'params': param, 'weight_decay': quant_params['weight decay']}
        else:
            p = {'params': param, 'weight_decay': 0.}
        param_group.append(p)

    # Optimization:
    learning_rate = quant_params['learning rate']
    step_size = quant_params['scheduler step'] * quant_params['dataset splits']

    n_epochs = train_params['epochs'] * train_params['dataset splits']
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(param_group, lr=learning_rate)#설정 다시보기
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=quant_params['scheduler gamma'])
    
    # Train the original model
    fit_model(quantized_model, data_loaders, model_params['channels'], criterion, optimizer, lr_scheduler, cuda_device,
              n_epochs, val_params['frequency'], quant_params['checkpoint path'], model_filename)


    # ⑨ 모델을 다시 CPU 상태로 두고 QAT가 적용된 floating point 모델을 quantized integer model로 변환합니다. 
    print("===== Training Done =====")  
    quantized_model.to("cpu:0")    

    # Using high-level static quantization wrapper
    # The above steps, including torch.quantization.prepare, calibrate_model, and torch.quantization.convert, are also equivalent to
    # quantized_model = torch.quantization.quantize_qat(model=quatized_model, run_fn=train_model, run_args=[train_loader, test_loader, cuda_device], mapping=None, inplace=False)

    # quantized interger model 확인
    print("===== 8. Convert the quantized integer model  =====\n") 
    quantized_model = torch.quantization.convert(quantized_model, inplace=True)
    quantized_model.eval()
    torch.save(quantized_model.state_dict(), os.path.join(model_dir, quantized_model_filename))

    #quantized model latency 확인
    int7_cpu_inference_latency = measure_inference_latency(model= quantized_model, device=cpu_device, input_size=(1,3,32,32), num_samples=100)
    fp32_cpu_inference_latency = measure_inference_latency(model=model, device=cpu_device, input_size=(1,3,32,32), num_samples=100)
    fp32_gpu_inference_latency = measure_inference_latency(model=model, device=cuda_device, input_size=(1,3,32,32), num_samples=100)
    
    print("INT7 CPU Inference Latency: {:.2f} ms / sample".format(int7_cpu_inference_latency * 1000))
    print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(fp32_cpu_inference_latency * 1000))
    print("FP32 CUDA Inference Latency: {:.2f} ms / sample".format(fp32_gpu_inference_latency * 1000))



if __name__ == '__main__':
    main()
    
