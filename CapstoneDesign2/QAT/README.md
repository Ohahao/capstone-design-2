## QAT code 폴더
--
1. 8bit QAT(완료)
   - 실행 방법: main_train_qat8.py를 실행하면 saved_models에 QAT가 완료된 파일이 저장된다. 파일명은 main_train_qat8.py의 quantized_model_filename에서 설정한다.
   - 성능
     - PSNR/SSIM: 18.05/0.8954 (기준 Pipeline과 비교한 결과)
     - Inference Latency: 82.71ms (56.77MB)
     - Model Size: 40.22MB (기존 모델: 158.79MB)
       
2. 다른 비트 QAT(진행 중)
   



