import numpy as np
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def calculate_psnr_ssim(image_path_1, image_path_2):
    # 이미지를 읽어들입니다.
    img1 = io.imread(image_path_1)
    img2 = io.imread(image_path_2)

    # PSNR을 계산합니다.
    psnr = compare_psnr(img1, img2)

    # SSIM을 계산합니다. multichannel=True를 설정하여 컬러 이미지로 처리합니다.
    ssim = compare_ssim(img1, img2, multichannel=True)

    return psnr, ssim

# 이미지 경로 설정
image_path_1 = '/home/hyoh/Pipeline/python/result/image_w_rdu_161056.png'
image_path_2 = '/home/hyoh/Pipeline/python/result/image_w_rdu_qat12_161056.png'

# PSNR과 SSIM 계산
psnr, ssim = calculate_psnr_ssim(image_path_1, image_path_2)

print(f"PSNR: {psnr}")
print(f"SSIM: {ssim}")
