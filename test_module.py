from mamba_ssm import Mamba2, Mamba
import torch

batch, length, dim = 8, 64, 32

x = torch.randn(batch, length, dim).to("cuda")

# Define Mamba2 model
model = Mamba2(
    d_model=dim,  # Model dimension
    d_state=64,   # SSM state expansion factor
    d_conv=4,     # Local convolution width
    expand=2,     # Block expansion factor
).to("cuda")

# Forward pass
y = model(x)  # NO transpose needed

# Ensure output has the same shape
assert y.shape == x.shape

print("✅ Model ran successfully!")
