import torch
import functools
import os
from os.path import join
from ptflops import get_model_complexity_info

from model_qat import RDUNet
from model_qat import RDUNet_quant
from utils import set_seed


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


def load_torchscript_model(model_filepath, device):

    model = torch.jit.load(model_filepath, map_location=device)

    return model

def load_model(model, model_filepath, device):

    model.load_state_dict(torch.load(model_filepath, map_location=device))

    return model

def main():
    # 경로 및 장치 지정
    cpu_device = torch.device("cpu:0")
    cuda_device = torch.device("cuda:1")

    model_dir = "saved_models"
    model_filename = "model_color.pth"
    quantized_model_filename = "model_color_quant.pth"
    quantized_model_jit_filename = "model_color_quant.pt"
    model_filepath = os.path.join(model_dir, model_filename)
    quantized_model_filepath = os.path.join(model_dir, quantized_model_filename)
    quantized_model_jit_filepath = os.path.join(model_dir, quantized_model_jit_filename)

    # 저장된 model load해서 latency 확인
    print("====== 저장된 model load =======")

    # Load FP32 model
    fp32_model = RDUNet(channels=3, base_filters=128)
    fp32_model = load_model(model=fp32_model, model_filepath=model_filepath, device=cuda_device)
    fp32_model.to(cpu_device)

    # Load quantized model
    quantized_model = load_model(model=fp32_model, model_filepath=quantized_model_filepath, device=cuda_device)
    quantized_model.to(cpu_device)

    # Load quantized JIT model
    quantized_jit_model = load_torchscript_model(model_filepath=quantized_model_jit_filepath, device=cpu_device)

    print("====== load된 model latency 계산 =======")
    fp32_cpu_inference_latency = measure_inference_latency(
        model=model, device=cpu_device, input_size=(
            1, 3, 32, 32), num_samples=100)
    int8_cpu_inference_latency = measure_inference_latency(
        model=quantized_model, device=cpu_device, input_size=(
            1, 3, 32, 32), num_samples=100)
    int8_jit_cpu_inference_latency = measure_inference_latency(
        model=quantized_jit_model, device=cpu_device, input_size=(
            1, 3, 32, 32), num_samples=100)
    fp32_gpu_inference_latency = measure_inference_latency(
        model=model, device=cuda_device, input_size=(
            1, 3, 32, 32), num_samples=100)

    print(
        "FP32 CPU Inference Latency: {:.2f} ms / sample".format(fp32_cpu_inference_latency * 1000))
    print(
        "FP32 CUDA Inference Latency: {:.2f} ms / sample".format(fp32_gpu_inference_latency * 1000))
    print(
        "INT8 CPU Inference Latency: {:.2f} ms / sample".format(int8_cpu_inference_latency * 1000))
    print("INT8 JIT CPU Inference Latency: {:.2f} ms / sample".format(
        int8_jit_cpu_inference_latency * 1000))


if __name__ == '__main__':
    main()
