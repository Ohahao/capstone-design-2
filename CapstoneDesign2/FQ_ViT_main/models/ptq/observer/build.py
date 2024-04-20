# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
from .ema import EmaObserver
from .minmax import MinmaxObserver
from .omse import OmseObserver
from .percentile import PercentileObserver
from .ptf import PtfObserver

#observer 형식을 선택한다. 
str2observer = {
    'minmax': MinmaxObserver,
    'ema': EmaObserver,
    'omse': OmseObserver,
    'percentile': PercentileObserver,
    'ptf': PtfObserver
}

#observer 인스턴스를 생성한다. 
def build_observer(observer_str, module_type, bit_type, calibration_mode):
    observer = str2observer[observer_str]
    return observer(module_type, bit_type, calibration_mode)
