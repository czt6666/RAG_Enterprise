import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))

# 列出所有 GPU
for i in range(torch.cuda.device_count()):
    print(f"设备 {i}: {torch.cuda.get_device_name(i)}")

