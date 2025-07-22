import torch

# --- Setup ---
print("Streamed execution profiling...")
size = 1000
ITERS = 5

# Use pinned memory for faster async copies
A = torch.ones(size, size, device="cpu")
collected_data = [torch.empty(size, size, device="cpu") for _ in range(ITERS)]

# Setup CUDA streams
streams = []
for i in range(ITERS):
    streams.append(torch.cuda.Stream())

def streamed_data_processing():
    """Streamed processing - pipeline host-to-GPU, kernel, GPU-to-host operations"""
    
    # Queue all operations across streams without waiting
    for i in range(ITERS):
        with torch.cuda.stream(streams[i]):
            # Push range for current iteration
            torch.cuda.nvtx.range_push(f"stream_{i}_iteration")
            
            # Host to GPU transfer (non-blocking)
            torch.cuda.nvtx.range_push(f"stream_{i}_host_to_gpu")
            A_gpu = A.to("cuda", non_blocking=True)
            torch.cuda.nvtx.range_pop()
            
            # Kernel computation
            torch.cuda.nvtx.range_push(f"stream_{i}_kernel_computation")
            C_gpu = torch.mm(A_gpu, A_gpu)
            # for _ in range(10):
            #     C_gpu += torch.mm(A_gpu, A_gpu)
            torch.cuda.nvtx.range_pop()
            
            # GPU to host transfer (non-blocking)
            torch.cuda.nvtx.range_push(f"stream_{i}_gpu_to_host")
            collected_data[i].copy_(C_gpu, non_blocking=True)
            torch.cuda.nvtx.range_pop()
            
            # Pop iteration range
            torch.cuda.nvtx.range_pop()
    
    # Wait for all streams to complete
    torch.cuda.nvtx.range_push("stream_synchronization")
    for stream in streams:
        stream.synchronize()
    torch.cuda.nvtx.range_pop()

# Warmup iterations (not profiled)
warmup_iters = 2
print("Running warmup iterations...")
for _ in range(warmup_iters):
    streamed_data_processing()

# Start profiling
print("Starting profiled iterations...")
torch.cuda.cudart().cudaProfilerStart()

# Profiled iterations
torch.cuda.nvtx.range_push("streamed_processing")
streamed_data_processing()
torch.cuda.nvtx.range_pop()

# Stop profiling
torch.cuda.cudart().cudaProfilerStop()

print("Streamed profiling complete.")
print("Run with: nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu --capture-range=cudaProfilerApi --stop-on-range-end=true --cudabacktrace=true -x true -o streamed_profile python streamed_main.py")