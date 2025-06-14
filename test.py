import torch
import time

def time_contiguous_call(tensor, num_iterations, tensor_description, device):
    # Warm-up iterations
    for _ in range(min(100, num_iterations // 10)):
        _ = tensor.contiguous()
        if device.type == 'cuda':
            torch.cuda.synchronize()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    for _ in range(num_iterations):
        # The actual operation being timed
        contig_tensor = tensor.contiguous()
        # We don't need to do anything with contig_tensor for timing the call itself

    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    print(f"--- Timing .contiguous() on {tensor_description} tensor ---")
    print(f"Device: {tensor.device}")
    print(f"Original tensor is contiguous: {tensor.is_contiguous()}")
    print(f"Shape: {tensor.shape}, Dtype: {tensor.dtype}")
    print(f"Average time for .contiguous() over {num_iterations} iterations: {avg_time:.9f} seconds")
    # To be absolutely sure the output of .contiguous() is contiguous if work was done
    # print(f"Output tensor is contiguous: {contig_tensor.is_contiguous()}")
    print("-" * 40)
    return avg_time

# --- Parameters ---
batch_size = 256
features1 = 1024
features2 = 2048 # For creating a non-contiguous tensor via transpose
num_iterations = 1000

# --- Test on CPU ---
print("======== TESTING ON CPU ========")
device_cpu = torch.device("cpu")

# 1. On an already contiguous tensor (CPU)
contiguous_tensor_cpu = torch.randn(batch_size, features1, device=device_cpu)
assert contiguous_tensor_cpu.is_contiguous()
time_contiguous_call(contiguous_tensor_cpu, num_iterations, "already contiguous", device_cpu)

# 2. On a non-contiguous tensor (CPU) - created by transpose
# Ensure features1 != features2 to make transpose reliably non-contiguous
non_contiguous_tensor_cpu = torch.randn(features2, batch_size, device=device_cpu).T
assert not non_contiguous_tensor_cpu.is_contiguous()
assert non_contiguous_tensor_cpu.shape == (batch_size, features2)
time_contiguous_call(non_contiguous_tensor_cpu, num_iterations, "non-contiguous (transposed)", device_cpu)

# 3. On a non-contiguous tensor (CPU) - created by slicing
temp_slice_cpu = torch.randn(batch_size, features1 * 2, device=device_cpu)
non_contiguous_sliced_cpu = temp_slice_cpu[:, ::2]
assert not non_contiguous_sliced_cpu.is_contiguous()
assert non_contiguous_sliced_cpu.shape == (batch_size, features1)
time_contiguous_call(non_contiguous_sliced_cpu, num_iterations, "non-contiguous (sliced)", device_cpu)


# --- Test on GPU (if available) ---
if torch.cuda.is_available():
    print("\n======== TESTING ON GPU ========")
    device_gpu = torch.device("cuda")

    # 1. On an already contiguous tensor (GPU)
    contiguous_tensor_gpu = torch.randn(batch_size, features1, device=device_gpu)
    assert contiguous_tensor_gpu.is_contiguous()
    time_contiguous_call(contiguous_tensor_gpu, num_iterations, "already contiguous", device_gpu)

    # 2. On a non-contiguous tensor (GPU) - created by transpose
    non_contiguous_tensor_gpu = torch.randn(features2, batch_size, device=device_gpu).T
    assert not non_contiguous_tensor_gpu.is_contiguous()
    assert non_contiguous_tensor_gpu.shape == (batch_size, features2)
    time_contiguous_call(non_contiguous_tensor_gpu, num_iterations, "non-contiguous (transposed)", device_gpu)

    # 3. On a non-contiguous tensor (GPU) - created by slicing
    temp_slice_gpu = torch.randn(batch_size, features1 * 2, device=device_gpu)
    non_contiguous_sliced_gpu = temp_slice_gpu[:, ::2]
    # It's possible for some slice operations on GPU to result in contiguous tensors
    # if PyTorch's memory manager can optimize it or if the slice is trivial.
    # We'll print its status.
    print(f"(Note: Sliced GPU tensor is_contiguous: {non_contiguous_sliced_gpu.is_contiguous()})")
    if not non_contiguous_sliced_gpu.is_contiguous(): # Only time if it's actually non-contiguous
         time_contiguous_call(non_contiguous_sliced_gpu, num_iterations, "non-contiguous (sliced)", device_gpu)
    else:
        print("Skipping timing for sliced GPU tensor as it was already contiguous.")

else:
    print("\nCUDA not available, skipping GPU tests.")