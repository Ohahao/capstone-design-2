# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
from .log2 import Log2Quantizer
from .uniform import UniformQuantizer

#양자화 방식 선택(균등 양자화 vs 로그 스케일 양자화)
str2quantizer = {'uniform': UniformQuantizer, 'log2': Log2Quantizer}


#양자화 인스턴스를 생성하는 함수이다. 
def build_quantizer(quantizer_str, bit_type, observer, module_type):
    quantizer = str2quantizer[quantizer_str]
    return quantizer(bit_type, observer, module_type)
