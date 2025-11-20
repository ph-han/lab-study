import torch

print("Pytorch version : ", torch.__version__)
print("CUDA is available? : ", torch.cuda.is_available())
print("GPU name : ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# 1. make tensor
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print("x: ", x)
print("shape: ", x.shape)

# 2. move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
print("device: ", x.device)

# 3. autograd: y = (x^2).sum()
x = torch.randn(3, 3, device=device, requires_grad=True) # gradient tracking
y = (x**2).sum()
print("y: ", y.item())

# 4. backward
y.backward()
print("x.grad: ", x.grad)

