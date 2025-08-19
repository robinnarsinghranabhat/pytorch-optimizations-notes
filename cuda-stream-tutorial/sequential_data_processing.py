import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pin-memory', action='store_true', default=True, help='Use pinned memory (default: True)')
parser.add_argument('--profile', action='store_true', default=True, help='Enable profiling (default: True)')
args = parser.parse_args()

PIN_MEMORY = args.pin_memory
PROFILE = args.profile

print(f"PROGRAM ARGS : (pin_memory={PIN_MEMORY}, profile={PROFILE})")
size = 256
ITERS = 50

input_shape = (64, size, size)
input = []
for i in range(ITERS):
    input.append(torch.randn(input_shape, device="cpu"))
if PIN_MEMORY:
    input = [in_.pin_memory() for in_ in input]

def sequential_data_processing():
    """
    Sequential version: No pipelining, no CUDA streams.
    Each input is processed fully before moving to the next.
    """
    torch.cuda.synchronize()
    for i in range(ITERS):
        torch.cuda.nvtx.range_push(f"seq_{i}_iteration")
        # Data Transfer from Host-Memory to Device-Memory
        torch.cuda.nvtx.range_push(f"seq_{i}_host_to_gpu")
        A_gpu = input[i].to("cuda", non_blocking=False)
        torch.cuda.nvtx.range_pop()

        # Some Computation on that Data
        torch.cuda.nvtx.range_push(f"seq_{i}_kernel_computation")
        C_gpu = torch.matmul(A_gpu, A_gpu)
        for i in range(10):
            C_gpu = C_gpu + torch.matmul(A_gpu, A_gpu)
        torch.cuda.nvtx.range_pop()

        # Transfer Processed Data back to Host-Memory
        torch.cuda.nvtx.range_push(f"seq_{i}_gpu_to_host")
        input[i].copy_(C_gpu, non_blocking=False)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

def main():
    warmup_iters = 2
    print("Running warmup iterations...")
    for _ in range(warmup_iters):
        sequential_data_processing()

    print("Starting profiled iterations...")
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("sequential_processing")
    sequential_data_processing()
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    print("Sequential profiling complete.")

if __name__ == "__main__":
    main()