import yaml
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
import torch.optim as optim
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

from model import RDUNet
from model import RDUNet_qaunt
from data_management import NoisyImagesDataset, DataSampler
from train import fit_model
from transforms import AdditiveWhiteGaussianNoise, RandomHorizontalFlip, RandomVerticalFlip, RandomRot90
from utils import set_seed

def load_model(model, model_filepath, device):

    model.load_state_dict(torch.load(model_filepath, map_location=device))

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

def main():
    
    with open('config.yaml', 'r') as stream:                # Load YAML configuration file.
        config = yaml.safe_load(stream)

    model_params = config['model']
    train_params = config['train']
    val_params = config['val']

    # Defining model:
    set_seed(0)
    model = RDUNet(**model_params)

    '''
    print('Model summary:')
    test_shape = (model_params['channels'], train_params['patch size'], train_params['patch size'])
    with torch.no_grad():
        macs, params = get_model_complexity_info(model, test_shape, as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    '''

    # Define the model name and use multi-GPU if it is allowed.
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")
    
    model_dir = "model_file"
    model_filename = "model_color.pth"
    quantized_model_filename = "model_color_quant.pth"
    model_filepath = os.path.join(model_dir, model_filename)
    
    model = load_model(model = model, model_filepath = model_filepath, device = cuda_device)
    
    #cpu에서 보낸 후 학습모드로 변경
    model.to(cpu_device)
    fused_model = copy.deepcopy(model) #layer fusion을 위한 model copy

    model.train()
    fused_model.train()
    
    # Fuse the model in place rather manually.
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
                    
    # print FP32 model
    print(model)
    # print fused model
    print(fused_model)

    # layer fusion 적용 환인 
    # Model and fused model should be equivalent.
    model.eval()
    fused_model.eval()
    assert model_equivalence(model_1=model, model_2=fused_model, device=cpu_device, rtol=1e-03, atol=1e-06, num_tests=100, input_size=(1,3,32,32)), "Fused model is not equivalent to the original model!"

    #quantizaion model 불러오기
    quantized_model = RDUNet_qaunt(model_fp32=fused_model)

    # quantizaion configuration 설정
    #quantization_config = torch.quantization.get_default_qat_qconfig("fbgemm") #나중에 고르기
    
    # Quantization Configuration
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
    print(quantized_model.qconfig)
    
    torch.quantization.prepare_qat(quantized_model, inplace=True)

    # training data를 활용한 calibration 진행
    print("Training QAT Model...")
    quantized_model.train()

    param_group = []
    for name, param in quantized_model.named_parameters():
        if 'conv' in name and 'weight' in name:
            p = {'params': param, 'weight_decay': train_params['weight decay']}
        else:
            p = {'params': param, 'weight_decay': 0.}
        param_group.append(p)

    # Load training and validation file names.
    # Modify .txt files if datasets do not fit in memory.
    with open('train_files.txt', 'r') as f_train, open('val_files.txt', 'r') as f_val:
        raw_train_files = f_train.read().splitlines()
        raw_val_files = f_val.read().splitlines()
        train_files = list(map(lambda file: join(train_params['dataset path'], file), raw_train_files))
        val_files = list(map(lambda file: join(val_params['dataset path'], file), raw_val_files))

    training_transforms = transforms.Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRot90()
    ])

    # Predefined noise level
    train_noise_transform = [AdditiveWhiteGaussianNoise(train_params['noise level'], clip=True)]
    val_noise_transforms = [AdditiveWhiteGaussianNoise(s, fix_sigma=True, clip=True) for s in val_params['noise levels']]

    print('\nLoading training dataset:')
    training_dataset = NoisyImagesDataset(train_files,
                                          model_params['channels'],
                                          train_params['patch size'],
                                          training_transforms,
                                          train_noise_transform)

    print('\nLoading validation dataset:')
    validation_dataset = NoisyImagesDataset(val_files,
                                            model_params['channels'],
                                            val_params['patch size'],
                                            None,
                                            val_noise_transforms)
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

    # Train the model
    fit_model(quantized_model, data_loaders, model_params['channels'], criterion, optimizer, lr_scheduler, cuda_device,
              n_epochs, val_params['frequency'], train_params['checkpoint path'], quantized_model_filename)


if __name__ == '__main__':
    main()
    
