## QAT model 성능 측정 방법
1. Pipeline/python의 demo2.py를 실행한다.
   - demo2.py 실행 전 수정 요소
     1) (demo2.py)output_image_path: 출력 이미지 이름을 지정한다.
     2) (pipeline.py) from (양자화 모델) import RDUNet, RDUNet_quant: 테스트할 비트수에 해당하는 모델을 불러와야한다. 이때 사용하는 모델 파일은 QAT 디렉토리에 위치한다.
       양자화 모델 파일명: model_qat4, model_qat6, model_qat8, model_qat10, model_qat12
     3) (pipeline.py) 모델 로드 코드 선택: 원본 모델 / 양자화 모델(8bit 이외) / 8bit QAT model 각각 모델을 로드하는 코드가 다르다. 필요에 따라 주석처리를 하여 사용한다.
        - 원본 모델![Uploading 스크린샷 2024-05-18 오후 2.18.04.png…]()
        - 양자화 모델
          #======== 양자화 모델 적용 ==========#
          #Sub-8bit Quantized 모델 load
          model_fp32 = RDUNet(last_calibrate=False, quant=False, calibrate=False, convert=False, device=cuda_device, **model_params)
          quantized_model = RDUNet_quant(model_fp32, device=cuda_device)
          #state_dict key의 차원 맞춰주기
          state_dict = torch.load(quantized_model_filepath)
          #.weight로 끝나는 키만 사용하여 모델의 가중치를 설정
          filtered_state_dict = {k: v for k, v in state_dict.items() if not k.endswith('.weight_quantized')}
          quantized_model.load_state_dict(filtered_state_dict, strict=False)
        - 8bit QAT model
          #======== 8bit QAT model 적용 =========# 
          #large model 로드
          model = RDUNet(channels=3, base_filters=64, device=cpu_device)
          model_fp32 = RDUNet(channels=3, base_filters=64, device=cpu_device)
          quantized_model = RDUNet_quant(model_fp32)
          model.load_state_dict(torch.load(model_filepath, map_location=cuda_device), strict=False)
      
          
          #cpu 사용!!
          #양자화 설정
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
          quantized_model = torch.quantization.prepare(quantized_model, inplace=True)
          quantized_model = torch.quantization.convert(quantized_model, inplace=True)   
          
          #양자화 모델 인스턴스화
          #quantized_model = RDUNet_quant(model_fp32)
          quantized_model.load_state_dict(torch.load(quantized_model_filepath), strict=False)
      5) (pipeline.py) inference model 선택: inference에 사용할 모델 이름을 선택한다. 원본 모델은 model_fp32, 양자화 & 8bit QAT 모델은 quantized_model을 사용한다.
      6) (pipeline.py) device 선택: inference 시 사용할 device를 선택한다. 8bit QAT만 cpu_device를 사용하고 나머지 2경우는 cuda_device를 사용한다.
    2. demo2.py 출력 이미지를 psnr.py에 입력하여 psnr을 측정한다. 
