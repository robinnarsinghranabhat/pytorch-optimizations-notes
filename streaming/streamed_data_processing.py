"""

Understanding on CPU and GPU interaction to optimize pytorch programs.

CPU can delagates work to GPU in following ways, to free itself do other stuffs.  
CPU instruction that calls CUDA-API to initiate :
1. Host-to-Memory Transfer
2. GPU Kernel Execution
3. Memory-to-Host Transfer

Each of these operations are independent (aynschronous ?). For example, in code below :
```

A_gpu = torch.zeroes((64,64), device="cpu", pin_memory=True) # Executed in CPU

# CPU executes a CUDA-API `cudaMemCpyAsync` call for Data Transfer. DMA Engine takes over data transfer now.
# Meanwhile, CPU can continue executing next instruction below while that transfer happens. 
A_gpu.to("cuda", non_blocking=True) 

# CPU side instruction to allocate some tensors. 
B_gpu = torch.zeroes((64,64), device="cpu")

# Suppose B_gpu allocation completed but `Host-to-Memory Transfer` is still ongoing.

# Here, CPU executes a CUDA-API `cudaMemCpyAsync` and delagates the Data Transfer task to DMA engine.
# But as previous data transfer of A_gpu is ongoing, this task sits in the Queue. But CPU is free and can continue.
B_gpu = B_gpu.to("cuda", non_blocking=True) 

# CPU is blocked by this instruction
cpu_variables = [ i**2 for i in range(100) ]

# Assume A_gpu transferred has finished. In the background, DMA will begin Host-to-device transfer of B_gpu now.
# Meanwhile, CPU executes instruction to launch a matmul kernel in GPU
# Program execution returns back to CPU and CPU continues
A_gpu = torch.matmul(A_gpu, A_gpu)

# At this point, say,  "Host-to-device TRansfer" of B_gpu and "Mat-mul operation of A_gpu" are happening at the same time.

```


In the program below, we show how to levarage this pipelining for all above three operations.

Running profiler : 
nsys profile --force-overwrite true  -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu --capture-range=cudaProfilerApi --cudabacktrace=true -x true -o test python streamed_processing.py
"""

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

## Prepare The Input to be Processed
## Suppose we want 5 independent inputs to be processed.
## Ideally, this example would be more relavant if we wanted different kind of processing for each of these input. 
## For example : apply softmax for first one, apply conv. to second e.t.c
## But to sake of brevity, we apply same processing to each of them.
## If you are still ruming on why not arrage whole input as a single Batch and continue.
## A practical scenario is when whole input is just too big to fit under GPU for processing :)
  
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
    