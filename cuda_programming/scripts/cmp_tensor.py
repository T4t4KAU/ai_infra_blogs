import numpy as np

m, n = 64, 128

O_torch = np.fromfile("./data/O.bin", dtype=np.float32).reshape(m, n)
O_cuda  = np.fromfile("./output/O_cuda.bin", dtype=np.float32).reshape(m, n)

# Compute error
diff = O_torch - O_cuda
max_error = np.abs(diff).max()
mse = (diff ** 2).mean()

print("🔍 Comparison Result:")
print(f"Max absolute error: {max_error:.6e}")
print(f"MSE: {mse:.6e}")

assert max_error < 1e-5, "❌ CUDA output differs too much from PyTorch!"
print("\n✅ PASSED: CUDA and PyTorch outputs are numerically close.")