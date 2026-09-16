import torch
import time

# ====== CONFIG ======
target_vram_usage = 0.8  # 80% VRAM usage
matrix_size = 4096  # Controls compute load (increase for more GPU usage)
device = "cuda"

# ====================

assert torch.cuda.is_available(), "CUDA is not available!"

gpu = torch.cuda.get_device_properties(0)
total_vram = gpu.total_memory

target_bytes = int(total_vram * target_vram_usage)

print(f"Total VRAM: {total_vram / 1e9:.2f} GB")
print(f"Target VRAM: {target_bytes / 1e9:.2f} GB")

# Allocate memory progressively
buffers = []
allocated = 0

while allocated < target_bytes:
    try:
        # Allocate ~256MB chunks
        chunk = torch.empty((256, 1024, 256), dtype=torch.float32, device=device)
        buffers.append(chunk)
        allocated += chunk.numel() * 4
        print(f"Allocated: {allocated / 1e9:.2f} GB", end="\r")
    except RuntimeError:
        print("\nReached allocation limit.")
        break

# Create matrices for compute stress
a = torch.randn((matrix_size, matrix_size), device=device)
b = torch.randn((matrix_size, matrix_size), device=device)

try:
    while True:
        c = torch.matmul(a, b)
        _ = c.sum().item()

except KeyboardInterrupt:
    print("\nStopped by user.")
