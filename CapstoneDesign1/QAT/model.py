import torch
import torch.nn as nn

from torch.nn.quantized import FloatFunctional
from typing import List
 

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


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):      
        super(DownsampleBlock, self).__init__()
        in_channels = int(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.actv = nn.PReLU(out_channels)

    def forward(self, x):
        #텐서가 FP에서 quantized model로 양자화된다. 
        return self.actv(self.conv(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, cat_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.f_add = FloatFunctional()

        add_cat = self.f_add.add(in_channels, cat_channels)
        self.conv = nn.Conv2d(add_cat, out_channels, 3, padding=1)
        self.conv_t = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.actv = nn.PReLU(out_channels)
        self.actv_t = nn.PReLU(in_channels)

    def forward(self, x: List[torch.Tensor]):
        #upsample, concat = x
        upsample = x[0]
        concat = x[1]
        upsample = self.actv_t(self.conv_t(upsample))
        return self.actv(self.conv(self.f_add.cat([concat, upsample], 1)))


class InputBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InputBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.actv_1 = nn.PReLU(out_channels)
        self.actv_2 = nn.PReLU(out_channels)


    def forward(self, x):
        #x = self.quant(x)
        x = self.actv_1(self.conv_1(x))
        return self.actv_2(self.conv_2(x))


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutputBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.actv_1 = nn.PReLU(in_channels)
        self.actv_2 = nn.PReLU(out_channels)


    def forward(self, x):
        x = self.actv_1(self.conv_1(x))
        return self.actv_2(self.conv_2(x))


class DenoisingBlock(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels):
        super(DenoisingBlock, self).__init__()
        self.f_add = FloatFunctional()
        self.conv_0 = nn.Conv2d(in_channels, inner_channels, 3, padding=1)
        self.conv_1 = nn.Conv2d(self.f_add.add(in_channels, inner_channels), inner_channels, 3, padding=1)
        in_channels_1 = int(self.f_add.add(self.f_add.mul_scalar(inner_channels, 2.0), in_channels))
        self.conv_2 = nn.Conv2d(in_channels_1, inner_channels, 3, padding=1)
        in_channels_2 = int(self.f_add.add(self.f_add.mul_scalar(inner_channels, 3.0), in_channels))
        self.conv_3 = nn.Conv2d(in_channels_2, out_channels, 3, padding=1)


        self.actv_0 = nn.PReLU(inner_channels)
        self.actv_1 = nn.PReLU(inner_channels)
        self.actv_2 = nn.PReLU(inner_channels)
        self.actv_3 = nn.PReLU(out_channels)


    def forward(self, x):
        out_0 = self.actv_0(self.conv_0(x))

        #out_0 = self.quant(out_0)
        out_0 = self.f_add.cat([x, out_0], 1)
        out_1 = self.actv_1(self.conv_1(out_0))

        #out_1 = self.quant(out_1)
        out_1 = self.f_add.cat([out_0, out_1], 1)
        out_2 = self.actv_2(self.conv_2(out_1))

        #out_2 = self.quant(out_2)
        out_2 = self.f_add.cat([out_1, out_2], 1)
        out_3 = self.actv_3(self.conv_3(out_2))
        out_3 = self.f_add.add(x, out_3)

        return out_3


class RDUNet(nn.Module):
    r"""
    Residual-Dense U-net for image denoising.
    """
    def __init__(self, **kwargs):
        super().__init__()
        
        #QuantStub: FP -> INT8
        #self.quant = torch.quantization.QuantStub()
        #self.dequant = torch.quantization.DeQuantStub()
        self.f_mul = FloatFunctional()

        channels = kwargs['channels']
        filters_0 = kwargs['base filters']
        filters_1 = int(self.f_mul.mul_scalar(filters_0, 2.0))
        filters_2 = int(self.f_mul.mul_scalar(filters_0, 4.0))
        filters_3 = int(self.f_mul.mul_scalar(filters_0, 8.0))

        # Encoder:
        # Level 0:
        self.input_block = InputBlock(channels, filters_0)
        self.block_0_0 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)
        self.block_0_1 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)
        self.down_0 = DownsampleBlock(filters_0, filters_1)

        # Level 1:
        self.block_1_0 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)
        self.block_1_1 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)
        self.down_1 = DownsampleBlock(filters_1, filters_2)

        # Level 2:
        self.block_2_0 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)
        self.block_2_1 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)
        self.down_2 = DownsampleBlock(filters_2, filters_3)

        # Level 3 (Bottleneck)
        self.block_3_0 = DenoisingBlock(filters_3, filters_3 // 2, filters_3)
        self.block_3_1 = DenoisingBlock(filters_3, filters_3 // 2, filters_3)

        # Decoder
        # Level 2:
        self.up_2 = UpsampleBlock(filters_3, filters_2, filters_2)
        self.block_2_2 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)
        self.block_2_3 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)

        # Level 1:
        self.up_1 = UpsampleBlock(filters_2, filters_1, filters_1)
        self.block_1_2 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)
        self.block_1_3 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)

        # Level 0:
        self.up_0 = UpsampleBlock(filters_1, filters_0, filters_0)
        self.block_0_2 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)
        self.block_0_3 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)

        self.output_block = OutputBlock(filters_0, channels)

        #DeQuantStub: INT8 -> FP
        #self.dequant = torch.ao.quantization.DeQuantStub()


    def forward(self, inputs):
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

        out_3 = self.down_2(out_2)          # Level 3 (Bottleneck)
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



class RDUNet_qaunt(nn.Module):
    def __init__(self, model_fp32):
        super(RDUNet_qaunt, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.model_fp32 = model_fp32    

    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x
