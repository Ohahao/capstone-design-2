import torch
import torch.nn as nn

from torch.nn.quantized import FloatFunctional
from typing import List
import os, sys
sys.path.append("/home/hyoh/QAT/FQ_ViT_main/models/ptq")
from layers import QAct, QConv2d, QConvTranspose2d, QLinear

from FQ_ViT_main.config import Config
 

@torch.no_grad()
def init_weights(init_type='xavier'):
    if init_type == 'xavier':
        init = nn.init.xavier_normal_
    elif init_type == 'he':
        init = nn.init.kaiming_normal_
    else:
        init = nn.init.orthogonal_

    def initializer(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            init(m.weight)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, 1.0, 0.01)
            nn.init.zeros_(m.bias)

    return initializer


class QPReLU(nn.Module):
    def __init__(self, device, num_parameters=1, init: float = 0.25):
        super(QPReLU, self).__init__()
        self.num_parameters = num_parameters
        self.weight = nn.Parameter(torch.Tensor(num_parameters).fill_(init))
        self.relu1 = nn.ReLU().to(device)  # 이 부분 수정
        self.relu2 = nn.ReLU().to(device)  # 이 부분 수정
        self.f_mul_neg_one1 = nnq.FloatFunctional().to(device)  # 이 부분 수정
        self.f_mul_neg_one2 = nnq.FloatFunctional().to(device)  # 이 부분 수정
        self.f_mul_alpha = nnq.FloatFunctional().to(device)  # 이 부분 수정
        self.f_add = nnq.FloatFunctional().to(device)  # 이 부분 수정
        self.quant = torch.quantization.QuantStub().to(device)  # 이 부분 수정
        self.dequant = torch.quantization.DeQuantStub().to(device)  # 이 부분 수정
        self.quant2 = torch.quantization.QuantStub().to(device)  # 이 부분 수정
        self.quant3 = torch.quantization.QuantStub().to(device)  # 이 부분 수정
        self.neg_one = torch.Tensor([-1.0]).to(device)  # 이 부분 수정
        self.device = device
    
    def forward(self, x):
        x = self.quant(x).to(self.device)
        
        # PReLU, with modules only
        x1 = self.relu1(x)     
        neg_one_q = self.quant2(self.neg_one).to(self.device)
        weight_q = self.quant3(self.weight).to(self.device)

        # 연산에 사용되는 모든 모듈을 self.device로 이동
        self.f_mul_neg_one2.to(self.device)
        self.relu2.to(self.device)
        self.f_mul_neg_one1.to(self.device)
        self.f_add.to(self.device)
        
        y = self.f_mul_neg_one2.mul(
                self.relu2(
                    self.f_mul_neg_one1.mul(x, neg_one_q),
                ),
            neg_one_q)   

        # weight_q 차원 변경 및 확장
        weight_q = weight_q.view(1, -1, 1, 1)  # [1, num_parameters, 1, 1]
        weight_q = weight_q.expand_as(y)  # [1, 64, 32, 32]로 확장
        x2 = self.f_mul_alpha.mul(weight_q, y)
        
        x = self.f_add.add(x1, x2)
        x = self.dequant(x).to(self.device)
        return x



class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, quant, calibrate, last_calibrate, convert, device):      
        super(DownsampleBlock, self).__init__()
        cfg = Config()
        in_channels = int(in_channels)
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.convert = convert
        self.device = device

        #self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = QConv2d(in_channels,
                            out_channels,
                            kernel_size=2,
                            stride=2,
                            quant=quant,
                            calibrate=calibrate,
                            last_calibrate=last_calibrate,
                            bit_type=cfg.BIT_TYPE_10_W,
                            calibration_mode=cfg.CALIBRATION_MODE_W,
                            observer_str=cfg.OBSERVER_W,
                            quantizer_str=cfg.QUANTIZER_W,
                            device=self.device)
        #self.actv = nn.PReLU(out_channels)
        self.actv = QAct( num_parameters=out_channels,
                          quant = quant,
                          calibrate = calibrate,
                          last_calibrate = last_calibrate,
                          bit_type=cfg.BIT_TYPE_10_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A,
                          device=self.device)

    def forward(self, x):
        self.conv.update_settings(quant=self.quant, calibrate=self.calibrate, last_calibrate=self.last_calibrate, convert=self.convert, device=self.device)
        #self.actv.update_settings(quant=self.quant, calibrate=self.calibrate, last_calibrate=self.last_calibrate, convert=self.convert, device=self.device)

        return self.actv(self.conv(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, cat_channels, out_channels, quant, calibrate, last_calibrate, convert, device):
        super(UpsampleBlock, self).__init__()
        cfg = Config()
        self.f_add = FloatFunctional()
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.convert = convert
        self.device = device

        add_cat = self.f_add.add(in_channels, cat_channels)
        #self.conv = nn.Conv2d(add_cat, out_channels, 3, padding=1)
        self.conv = QConv2d(add_cat,
                            out_channels,
                            kernel_size=3,
                            padding=1,
                            quant=quant,
                            calibrate=calibrate,
                            bit_type=cfg.BIT_TYPE_10_W,
                            calibration_mode=cfg.CALIBRATION_MODE_W,
                            observer_str=cfg.OBSERVER_W,
                            quantizer_str=cfg.QUANTIZER_W,
                            device=self.device) 
        #self.conv_t = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.conv_t = QConvTranspose2d( in_channels,
                                        in_channels,
                                        kernel_size=2,
                                        stride=2,
                                        bit_type=cfg.BIT_TYPE_10_W,
                                        calibration_mode=cfg.CALIBRATION_MODE_W,    #channel wise
                                        observer_str=cfg.OBSERVER_W,    #minmax
                                        quantizer_str=cfg.QUANTIZER_W,
                                        device=self.device)
        self.actv = nn.PReLU(out_channels)
        '''
        self.actv = QAct( num_parameters=out_channels,
                          quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_10_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A,
                          device=self.device)
        '''
        self.actv_t = nn.PReLU(in_channels)
        '''
        self.actv_t = QAct( num_parameters=in_channels,
                           quant=quant,
                           calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_10_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A,
                          device=self.device)
        '''


    def forward(self, x: List[torch.Tensor]):
        #Set quant & calibrate
        self.conv.update_settings(quant=self.quant, calibrate=self.calibrate, last_calibrate=self.last_calibrate, convert=self.convert, device=self.device)
        self.conv_t.update_settings(quant=self.quant, calibrate=self.calibrate, last_calibrate=self.last_calibrate, convert=self.convert, device=self.device)
        #self.actv.update_settings(quant=self.quant, calibrate=self.calibrate, last_calibrate=self.last_calibrate, convert=self.convert, device=self.device)
        #self.actv_t.update_settings(quant=self.quant, calibrate=self.calibrate, last_calibrate=self.last_calibrate, convert=self.convert, device=self.device)

        #upsample, concat = x
        upsample = x[0]
        concat = x[1]
        upsample = self.actv_t(self.conv_t(upsample)).to(self.device)
        return self.actv(self.conv(self.f_add.cat([concat, upsample], 1))).to(self.device)


class InputBlock(nn.Module):
    def __init__(self, in_channels, out_channels, quant, calibrate, last_calibrate, convert, device):
        super(InputBlock, self).__init__()
        cfg = Config()

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.device = device
        self.convert = convert
        
        #self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv_1 = QConv2d(in_channels,
                            out_channels,
                            kernel_size=3,
                            padding=1,
                            quant=self.quant,
                            calibrate=self.calibrate,
                            last_calibrate = last_calibrate,
                            bit_type=cfg.BIT_TYPE_10_W,
                            calibration_mode=cfg.CALIBRATION_MODE_W,
                            observer_str=cfg.OBSERVER_W,
                            quantizer_str=cfg.QUANTIZER_W,
                            device=self.device)
        #self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv_2 = QConv2d(out_channels,
                            out_channels,
                            kernel_size=3,
                            padding=1,
                            quant=self.quant,
                            calibrate=self.calibrate,
                            last_calibrate = last_calibrate,
                            bit_type=cfg.BIT_TYPE_10_W,
                            calibration_mode=cfg.CALIBRATION_MODE_W,
                            observer_str=cfg.OBSERVER_W,
                            quantizer_str=cfg.QUANTIZER_W,
                            device=self.device)
        self.actv_1 = nn.PReLU(out_channels)
        '''
        self.actv_1 = QAct( num_parameters=out_channels,
                            quant=self.quant,
                            calibrate=self.calibrate,
                            last_calibrate = last_calibrate,
                          bit_type=cfg.BIT_TYPE_10_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A,
                          device=self.device)
        '''
        self.actv_2 = nn.PReLU(out_channels)
        '''
        self.actv_2 = QAct( num_parameters=out_channels,
                            quant=self.quant,
                            calibrate=self.calibrate,
                            last_calibrate = last_calibrate,
                          bit_type=cfg.BIT_TYPE_10_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A,
                          device=self.device)
        '''


    def forward(self, x):
        #x = self.quant(x)
        #print("InputBlock quant: ", self.quant)
        #print("InputBlock calibrate: ", self.calibrate)

        #Set quant & calibrate
        self.conv_1.update_settings(quant=self.quant, calibrate=self.calibrate, last_calibrate=self.last_calibrate, convert=self.convert, device=self.device)
        self.conv_2.update_settings(quant=self.quant, calibrate=self.calibrate, last_calibrate=self.last_calibrate, convert=self.convert, device=self.device)
        #self.actv_1.update_settings(quant=self.quant, calibrate=self.calibrate, last_calibrate=self.last_calibrate, convert=self.convert, device=self.device)
        #self.actv_2.update_settings(quant=self.quant, calibrate=self.calibrate, last_calibrate=self.last_calibrate, convert=self.convert, device=self.device)

        x = self.actv_1(self.conv_1(x)).to(self.device)
        return self.actv_2(self.conv_2(x))


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_channels, quant, calibrate, last_calibrate, convert, device):
        super(OutputBlock, self).__init__()
        cfg = Config()
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.device = device
        self.convert = convert
        
        #self.conv_1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv_1 = QConv2d(in_channels,
                            in_channels,
                            kernel_size=3,
                            padding=1,
                            quant=quant,
                            calibrate=calibrate,
                            last_calibrate = last_calibrate,
                            bit_type=cfg.BIT_TYPE_10_W,
                            calibration_mode=cfg.CALIBRATION_MODE_W,
                            observer_str=cfg.OBSERVER_W,
                            quantizer_str=cfg.QUANTIZER_W,
                            device=self.device)
        #self.conv_2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv_2 = QConv2d(in_channels,
                            out_channels,
                            kernel_size=3,
                            padding=1,
                            quant=quant,
                            calibrate=calibrate,
                            last_calibrate = last_calibrate,
                            bit_type=cfg.BIT_TYPE_10_W,
                            calibration_mode=cfg.CALIBRATION_MODE_W,
                            observer_str=cfg.OBSERVER_W,
                            quantizer_str=cfg.QUANTIZER_W,
                            device=self.device)

        self.actv_1 = nn.PReLU(in_channels)
        '''
        self.actv_1 = QAct( num_parameters=in_channels,
                            quant=quant,
                            calibrate=calibrate,
                            last_calibrate = last_calibrate,
                          bit_type=cfg.BIT_TYPE_10_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A,
                          device=self.device)
        '''
        self.actv_2 = nn.PReLU(out_channels)
        '''
        self.actv_2 = QAct( num_parameters=out_channels,
                            quant=quant,
                            calibrate=calibrate,
                            last_calibrate = last_calibrate,
                          bit_type=cfg.BIT_TYPE_10_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A,
                          device=self.device)
        '''


    def forward(self, x):
        #Set quant & calibrate
        self.conv_1.update_settings(quant=self.quant, calibrate=self.calibrate, last_calibrate=self.last_calibrate, convert=self.convert, device=self.device)
        self.conv_2.update_settings(quant=self.quant, calibrate=self.calibrate, last_calibrate=self.last_calibrate, convert=self.convert, device=self.device)
        #self.actv_1.update_settings(quant=self.quant, calibrate=self.calibrate, last_calibrate=self.last_calibrate, convert=self.convert, device=self.device)
        #self.actv_2.update_settings(quant=self.quant, calibrate=self.calibrate, last_calibrate=self.last_calibrate, convert=self.convert, device=self.device)

        x = self.actv_1(self.conv_1(x)).to(self.device)
        return self.actv_2(self.conv_2(x)).to(self.device)


class DenoisingBlock(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels, quant, calibrate, last_calibrate, convert, device):
        super(DenoisingBlock, self).__init__()
        cfg = Config()
        self.f_add = FloatFunctional()
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.device = device
        self.convert = convert

        #self.conv_0 = nn.Conv2d(in_channels, inner_channels, 3, padding=1)
        self.conv_0 = QConv2d(in_channels,
                            inner_channels,
                            kernel_size=3,
                            padding=1,
                            quant=quant,
                            calibrate=calibrate,
                            last_calibrate = last_calibrate,
                            bit_type=cfg.BIT_TYPE_10_W,
                            calibration_mode=cfg.CALIBRATION_MODE_W,
                            observer_str=cfg.OBSERVER_W,
                            quantizer_str=cfg.QUANTIZER_W,
                            device=self.device)
        #self.conv_1 = nn.Conv2d(self.f_add.add(in_channels, inner_channels), inner_channels, 3, padding=1)
        self.conv_1 = QConv2d(self.f_add.add(in_channels, inner_channels),
                            inner_channels,
                            kernel_size=3,
                            padding=1,
                            quant=quant,
                            calibrate=calibrate,
                            last_calibrate = last_calibrate,
                            bit_type=cfg.BIT_TYPE_10_W,
                            calibration_mode=cfg.CALIBRATION_MODE_W,
                            observer_str=cfg.OBSERVER_W,
                            quantizer_str=cfg.QUANTIZER_W,
                            device=self.device)
        in_channels_1 = int(self.f_add.add(self.f_add.mul_scalar(inner_channels, 2.0), in_channels))
        #self.conv_2 = nn.Conv2d(in_channels_1, inner_channels, 3, padding=1)
        self.conv_2 = QConv2d(in_channels_1,
                            inner_channels,
                            kernel_size=3,
                            padding=1,
                            quant=quant,
                            calibrate=calibrate,
                            last_calibrate = last_calibrate,
                            bit_type=cfg.BIT_TYPE_10_W,
                            calibration_mode=cfg.CALIBRATION_MODE_W,
                            observer_str=cfg.OBSERVER_W,
                            quantizer_str=cfg.QUANTIZER_W,
                            device=self.device)
        in_channels_2 = int(self.f_add.add(self.f_add.mul_scalar(inner_channels, 3.0), in_channels))
        #self.conv_3 = nn.Conv2d(in_channels_2, out_channels, 3, padding=1)
        self.conv_3 = QConv2d(in_channels_2,
                            out_channels,
                            kernel_size=3,
                            padding=1,
                            quant=quant,
                            calibrate=calibrate,
                            last_calibrate = last_calibrate,
                            bit_type=cfg.BIT_TYPE_10_W,
                            calibration_mode=cfg.CALIBRATION_MODE_W,
                            observer_str=cfg.OBSERVER_W,
                            quantizer_str=cfg.QUANTIZER_W,
                            device=self.device)

        self.actv_0 = nn.PReLU(inner_channels)
        '''
        self.actv_0 = QAct( num_parameters=inner_channels,
                            quant=quant,
                            calibrate=calibrate,
                            last_calibrate = last_calibrate,
                          bit_type=cfg.BIT_TYPE_10_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A,
                          device=self.device)
        '''
        self.actv_1 = nn.PReLU(inner_channels)
        '''
        self.actv_1 = QAct( num_parameters=inner_channels,
                            quant=quant,
                            calibrate=calibrate,
                            last_calibrate = last_calibrate,
                          bit_type=cfg.BIT_TYPE_10_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A,
                          device=self.device)
        '''
        self.actv_2 = nn.PReLU(inner_channels)
        '''
        self.actv_2 = QAct( num_parameters=inner_channels,
                            quant=quant,
                            calibrate=calibrate,
                            last_calibrate = last_calibrate,
                          bit_type=cfg.BIT_TYPE_10_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A,
                          device=self.device)
        '''
        self.actv_3 = nn.PReLU(out_channels)
        '''
        self.actv_3 = QAct( num_parameters=out_channels,
                            quant=quant,
                            calibrate=calibrate,
                            last_calibrate = last_calibrate,
                          bit_type=cfg.BIT_TYPE_10_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A,
                          device=self.device)
        '''

    def forward(self, x):
        #Set quant & calibrate
        self.conv_0.update_settings(quant=self.quant, calibrate=self.calibrate, last_calibrate=self.last_calibrate, convert=self.convert, device=self.device)
        self.conv_1.update_settings(quant=self.quant, calibrate=self.calibrate, last_calibrate=self.last_calibrate, convert=self.convert, device=self.device)
        self.conv_2.update_settings(quant=self.quant, calibrate=self.calibrate, last_calibrate=self.last_calibrate, convert=self.convert, device=self.device)
        self.conv_3.update_settings(quant=self.quant, calibrate=self.calibrate, last_calibrate=self.last_calibrate, convert=self.convert, device=self.device)
        #self.actv_0.update_settings(quant=self.quant, calibrate=self.calibrate, last_calibrate=self.last_calibrate, convert=self.convert, device=self.device)
        #self.actv_1.update_settings(quant=self.quant, calibrate=self.calibrate, last_calibrate=self.last_calibrate, convert=self.convert, device=self.device)
        #self.actv_2.update_settings(quant=self.quant, calibrate=self.calibrate, last_calibrate=self.last_calibrate, convert=self.convert, device=self.device)
        #self.actv_3.update_settings(quant=self.quant, calibrate=self.calibrate, last_calibrate=self.last_calibrate, convert=self.convert, device=self.device)
        
        x = x.to(self.device)
        out_0 = self.actv_0(self.conv_0(x)).to(self.device)

        #out_0 = self.quant(out_0)
        out_0 = self.f_add.cat([x, out_0], 1)
        out_1 = self.actv_1(self.conv_1(out_0)).to(self.device)

        #out_1 = self.quant(out_1)
        out_1 = self.f_add.cat([out_0, out_1], 1).to(self.device)
        out_2 = self.actv_2(self.conv_2(out_1)).to(self.device)

        #out_2 = self.quant(out_2)
        out_2 = self.f_add.cat([out_1, out_2], 1)
        out_3 = self.actv_3(self.conv_3(out_2)).to(self.device)
        out_3 = self.f_add.add(x, out_3)

        return out_3


class RDUNet(nn.Module):
    r"""
    Residual-Dense U-net for image denoising.
    """
    def __init__(self, last_calibrate, quant, calibrate, convert, device, **kwargs):
        super().__init__()
        
        #QuantStub: FP -> INT8
        #self.quant = torch.quantization.QuantStub()
        #self.dequant = torch.quantization.DeQuantStub()
        self.f_mul = FloatFunctional()

        channels = kwargs['channels']
        filters_0 = kwargs['base_filters']
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.convert = convert
        self.device = device
        
        filters_1 = int(self.f_mul.mul_scalar(filters_0, 2.0))
        filters_2 = int(self.f_mul.mul_scalar(filters_0, 4.0))
        filters_3 = int(self.f_mul.mul_scalar(filters_0, 8.0))

        # Encoder:
        # Level 0:
        self.input_block = InputBlock(channels, filters_0, self.quant, self.calibrate, self.last_calibrate, self.convert, self.device)
        self.block_0_0 = DenoisingBlock(filters_0, filters_0 // 2, filters_0, self.quant, self.calibrate, self.last_calibrate, self.convert, self.device)
        self.block_0_1 = DenoisingBlock(filters_0, filters_0 // 2, filters_0, self.quant, self.calibrate, self.last_calibrate, self.convert, self.device)
        self.down_0 = DownsampleBlock(filters_0, filters_1, self.quant, self.calibrate, self.last_calibrate, self.convert, self.device)

        # Level 1:
        self.block_1_0 = DenoisingBlock(filters_1, filters_1 // 2, filters_1, self.quant, self.calibrate, self.last_calibrate, self.convert, self.device)
        self.block_1_1 = DenoisingBlock(filters_1, filters_1 // 2, filters_1, self.quant, self.calibrate, self.last_calibrate, self.convert, self.device)
        self.down_1 = DownsampleBlock(filters_1, filters_2, self.quant, self.calibrate, self.last_calibrate, self.convert, self.device)

        # Level 2:
        self.block_2_0 = DenoisingBlock(filters_2, filters_2 // 2, filters_2, self.quant, self.calibrate, self.last_calibrate, self.convert, self.device)
        self.block_2_1 = DenoisingBlock(filters_2, filters_2 // 2, filters_2, self.quant, self.calibrate, self.last_calibrate, self.convert, self.device)
        self.down_2 = DownsampleBlock(filters_2, filters_3, self.quant, self.calibrate, self.last_calibrate, self.convert, self.device)

        # Level 3 (Bottleneck)
        self.block_3_0 = DenoisingBlock(filters_3, filters_3 // 2, filters_3, self.quant, self.calibrate, self.last_calibrate, self.convert, self.device)
        self.block_3_1 = DenoisingBlock(filters_3, filters_3 // 2, filters_3, self.quant, self.calibrate, self.last_calibrate, self.convert, self.device)

        # Decoder
        # Level 2:
        self.up_2 = UpsampleBlock(filters_3, filters_2, filters_2, self.quant, self.calibrate, self.last_calibrate, self.convert, self.device)
        self.block_2_2 = DenoisingBlock(filters_2, filters_2 // 2, filters_2, self.quant, self.calibrate, self.last_calibrate, self.convert, self.device)
        self.block_2_3 = DenoisingBlock(filters_2, filters_2 // 2, filters_2, self.quant, self.calibrate, self.last_calibrate, self.convert, self.device)

        # Level 1:
        self.up_1 = UpsampleBlock(filters_2, filters_1, filters_1, self.quant, self.calibrate, self.last_calibrate, self.convert, self.device)
        self.block_1_2 = DenoisingBlock(filters_1, filters_1 // 2, filters_1, self.quant, self.calibrate, self.last_calibrate, self.convert, self.device)
        self.block_1_3 = DenoisingBlock(filters_1, filters_1 // 2, filters_1, self.quant, self.calibrate, self.last_calibrate, self.convert, self.device)

        # Level 0:
        self.up_0 = UpsampleBlock(filters_1, filters_0, filters_0, self.quant, self.calibrate, self.last_calibrate, self.convert, self.device)
        self.block_0_2 = DenoisingBlock(filters_0, filters_0 // 2, filters_0, self.quant, self.calibrate, self.last_calibrate, self.convert, self.device)
        self.block_0_3 = DenoisingBlock(filters_0, filters_0 // 2, filters_0, self.quant, self.calibrate, self.last_calibrate, self.convert, self.device)

        self.output_block = OutputBlock(filters_0, channels, self.quant, self.calibrate, self.last_calibrate, self.convert, self.device)

        #DeQuantStub: INT8 -> FP
        #self.dequant = torch.ao.quantization.DeQuantStub()

    def set_quant_calibrate(self, quant, calibrate, last_calibrate, convert, device):
            # 모든 블록에 대해 quant와 calibrate 값을 설정합니다.
            self.quant = quant
            self.calibrate = calibrate
            self.last_calibrate = last_calibrate
            self.convert = convert
            self.device = device
            blocks = [
                self.input_block, self.output_block,self.block_0_0, self.block_0_1, self.block_0_2, self.block_0_3, self.block_1_0,
                self.block_1_1, self.block_1_2, self.block_1_3, self.block_2_0, self.block_2_1, self.block_2_2, self.block_2_3, self.block_3_0, self.block_3_1, self.up_0,
                self.up_1, self.up_2, self.down_0, self.down_1, self.down_2          
            ]
            # 각 블록에 대해 quant와 calibrate 값을 설정합니다.
            for block in blocks:
                if hasattr(block, 'quant'):
                    block.quant = quant
                if hasattr(block, 'calibrate'):
                    block.calibrate = calibrate
                if hasattr(block, 'last_calibrate'):
                    block.last_calibrate = last_calibrate
                if hasattr(block, 'convert'):
                    block.convert = convert
                if hasattr(block, 'device'):
                    block.device = device

    def forward(self, inputs):
        #print("quant: ", self.quant)
        #print("calibrate: ", self.calibrate)
        #inputs = self.quant(inputs)
        out_0 = self.input_block(inputs)    # Level 0
        out_0 = self.block_0_0(out_0)
        out_0 = self.block_0_1(out_0)

        out_1 = self.down_0(out_0)          # Level 1
        out_1 = self.block_1_0(out_1)
        out_1 = self.block_1_1(out_1)

        out_2 = self.down_1(out_1)          # Level 2
        out_2 = self.block_2_0(out_2)
        out_2 = self.block_2_1(out_2)

        out_3 = self.down_2(out_2)       # Level 3 (Bottleneck)
        out_3 = self.block_3_0(out_3)
        out_3 = self.block_3_1(out_3)

        out_4 = self.up_2([out_3, out_2])   # Level 2
        out_4 = self.block_2_2(out_4)
        out_4 = self.block_2_3(out_4)

        out_5 = self.up_1([out_4, out_1])   # Level 1
        out_5 = self.block_1_2(out_5)
        out_5 = self.block_1_3(out_5)

        out_6 = self.up_0([out_5, out_0])   # Level 0
        out_6 = self.block_0_2(out_6)
        out_6 = self.block_0_3(out_6)

        out = self.f_mul.add(self.output_block(out_6), inputs)
        #out = self.dequant(out)
        return out



class RDUNet_quant(nn.Module):
    def __init__(self, model_fp32, device, quant=False, calibrate=False, last_calibrate=False, convert=False):
        super(RDUNet_quant, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.model_fp32 = model_fp32    

    def set_quant_calibrate(self, quant, calibrate, last_calibrate, convert, device):
        # 내부 model_fp32의 설정 변경
        self.model_fp32.set_quant_calibrate(quant=quant, calibrate=calibrate, last_calibrate=last_calibrate, convert=convert, device=device)


    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x

        
