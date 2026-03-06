import math
import numpy as np
import torch


def self_attention_pytorch(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute naive single-head self-attention (forward only):
        O = softmax((Q @ K^T) / sqrt(d)) @ V

    Shapes:
        Q, K, V: [m, n] (row-major, float32 recommended)
        O:       [m, n]
    """
    if Q.dim() != 2 or K.dim() != 2 or V.dim() != 2:
        raise ValueError("Q, K, V must be 2D tensors of shape [m, n].")
    if Q.shape != K.shape or Q.shape != V.shape:
        raise ValueError(f"Q, K, V must have the same shape, got {Q.shape}, {K.shape}, {V.shape}.")

    m, n = Q.shape
    scale = 1.0 / math.sqrt(float(n))

    # [m, m]
    scores = torch.mm(Q, K.transpose(0, 1)) * scale

    # Row-wise softmax over last dimension
    attn = torch.softmax(scores, dim=-1)

    # [m, n]
    out = torch.mm(attn, V)
    return out


def save_tensor_bin(tensor: torch.Tensor, filename: str) -> None:
    """
    Save a tensor to a raw binary float32 file in row-major order.
    This matches a simple C/CUDA fread into float* buffer (no header).
    """
    arr = tensor.contiguous().cpu().numpy().astype(np.float32, copy=False)
    arr.tofile(filename)
    print(f"Saved: {filename} | shape={tuple(tensor.shape)} | dtype={arr.dtype} | bytes={arr.nbytes}")


def main() -> None:
    # -------------------------------
    # Configuration
    # -------------------------------
    m, n = 64, 128
    dtype = torch.float32
    seed = 42

    out_dir = "./data"
    q_path = f"{out_dir}/Q.bin"
    k_path = f"{out_dir}/K.bin"
    v_path = f"{out_dir}/V.bin"
    o_path = f"{out_dir}/O.bin"

    # If you want to compute on GPU, set use_cuda=True.
    # Note: saved binaries will still be written from CPU.
    use_cuda = False
    device = torch.device("cuda") if (use_cuda and torch.cuda.is_available()) else torch.device("cpu")

    # -------------------------------
    # Reproducibility
    # -------------------------------
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # -------------------------------
    # Generate inputs
    # -------------------------------
    Q = torch.randn(m, n, dtype=dtype, device=device)
    K = torch.randn(m, n, dtype=dtype, device=device)
    V = torch.randn(m, n, dtype=dtype, device=device)

    # -------------------------------
    # Run self-attention (forward)
    # -------------------------------
    with torch.no_grad():
        O = self_attention_pytorch(Q, K, V)

    # -------------------------------
    # Save to .bin (float32, row-major)
    # -------------------------------
    save_tensor_bin(Q, q_path)
    save_tensor_bin(K, k_path)
    save_tensor_bin(V, v_path)
    save_tensor_bin(O, o_path)

    print("Done.")


if __name__ == "__main__":
    main()