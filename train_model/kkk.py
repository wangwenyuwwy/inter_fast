import torch
print("PyTorch 版本:", torch.__version__)
print("PyTorch 编译的 CUDA 版本:", torch.version.cuda)
print("CUDA 是否可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("当前 GPU 的 CUDA 版本:", torch.cuda.get_device_properties(0).cuda_version)