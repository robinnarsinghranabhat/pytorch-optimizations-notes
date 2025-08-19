import torch
import argparse

# --- Argument Parsing ---
parser = argparse.ArgumentParser()
parser.add_argument('--pin-memory', action='store_true', default=True, help='Use pinned memory (default: True)')
parser.add_argument('--profile', action='store_true', default=True, help='Enable profiling (default: True)')

args = parser.parse_args()

PIN_MEMORY = args.pin_memory
PROFILE = args.profile

print(f"PROGRAM ARGS : (pin_memory={PIN_MEMORY}, profile={PROFILE}")
size = 256
ITERS = 50

  
input_shape = (64, size, size)
input = []
for i in range(ITERS):
    input.append(torch.randn(input_shape, device="cpu"))
if PIN_MEMORY:
    input = [in_.pin_memory() for in_ in input]


def streamed_data_processing():
    """
    Dummy Program to showwcase Pipelining to Speed-Up Procesing Time.
    -> Host-to-Memory Transfer
    -> GPU Kernel Execution
    -> Memory-to-Host Transfer
    """
    streams = [torch.cuda.Stream() for _ in range(ITERS)]

    # Before our Profiling starts, wait for CUDA API calls made before this step has completed.
    torch.cuda.synchronize()
    for i in range(ITERS):
        torch.cuda.nvtx.range_push(f"stream_{i}_iteration")
        with torch.cuda.stream(streams[i]):
            # Data Transfer from Host-Memory to Device-Memory
            torch.cuda.nvtx.range_push(f"stream_{i}_host_to_gpu")
            A_gpu = input[i].to("cuda", non_blocking=True)
            torch.cuda.nvtx.range_pop()

            # Some Computation on that Data
            torch.cuda.nvtx.range_push(f"stream_{i}_kernel_computation")
            # C_gpu = torch.matmul(A_gpu, A_gpu) + torch.matmul(A_gpu, A_gpu) + torch.matmul(A_gpu, A_gpu)
            C_gpu = torch.matmul(A_gpu, A_gpu)
            for i in range(10):
                C_gpu = C_gpu + torch.matmul(A_gpu, A_gpu)
            torch.cuda.nvtx.range_pop()

            # Transfer Processed Data back to Device-Memory
            torch.cuda.nvtx.range_push(f"stream_{i}_gpu_to_host")
            input[i].copy_(C_gpu, non_blocking=True)
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
    
    torch.cuda.nvtx.range_push("stream_synchronization")
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

def main():
        
    warmup_iters = 2
    print("Running warmup iterations...")
    for _ in range(warmup_iters):
        streamed_data_processing()

    
    print("Starting profiled iterations...")
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("streamed_processing")
    streamed_data_processing()
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    print("Streamed profiling complete.")

if __name__ == "__main__":
    main()
    