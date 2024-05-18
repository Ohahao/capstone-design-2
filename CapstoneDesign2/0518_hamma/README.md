## QAT model 성능 측정 방법
1. Pipeline/python의 demo2.py를 실행한다.
    1) (demo2.py)output_image_path: 출력 이미지 이름을 지정한다.
    2) (pipeline.py) from (양자화 모델) import RDUNet, RDUNet_quant: 테스트할 비트수에 해당하는 모델을 불러와야한다. 이때 사용하는 모델 파일은 QAT 디렉토리에 위치한다.
      양자화 모델 파일명: model_qat4, model_qat6, model_qat8, model_qat10, model_qat12
    3) (pipeline.py) 모델 로드 코드 선택: 원본 모델 / 양자화 모델(8bit 이외) / 8bit QAT model 각각 모델을 로드하는 코드가 다르다. 필요에 따라 주석처리를 하여 사용한다.
       - 원본 모델
  
         <img width="410" alt="스크린샷 2024-05-18 오후 2 19 01" src="https://github.com/Ohahao/capstone-design-2/assets/89395783/47e21945-637b-4ffd-870e-c076ae630b38">
  
        - 양자화 모델
     
       <img width="902" alt="스크린샷 2024-05-18 오후 2 19 45" src="https://github.com/Ohahao/capstone-design-2/assets/89395783/502340e9-1612-4bd1-8f94-c01fd8181fbd">
  
        - 8bit QAT model
        
          <img width="736" alt="스크린샷 2024-05-18 오후 2 20 27" src="https://github.com/Ohahao/capstone-design-2/assets/89395783/427bb8ad-692b-4eff-92f3-9fbdfcbefffb">
  
  
     5) (pipeline.py) inference model 선택: inference에 사용할 모델 이름을 선택한다. 원본 모델은 model_fp32, 양자화 & 8bit QAT 모델은 quantized_model을 사용한다.
     6) (pipeline.py) device 선택: inference 시 사용할 device를 선택한다. 8bit QAT만 cpu_device를 사용하고 나머지 2경우는 cuda_device를 사용한다.
2. demo2.py 출력 이미지를 psnr.py에 입력하여 psnr을 측정한다. 
