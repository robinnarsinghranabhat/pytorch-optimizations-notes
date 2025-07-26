import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--profile', action='store_true', default=True, help='Enable profiling')
args = parser.parse_args()

def cuda_interaction_demo():
    A_gpu = torch.zeros((8000, 8000), device="cpu", pin_memory=True)
    
    B_gpu = torch.ones((8000, 8000), device="cpu", pin_memory=True)
  
    torch.cuda.synchronize()


    torch.cuda.nvtx.range_push("1. DMA Transfer A to GPU")
    A_gpu = A_gpu.to("cuda", non_blocking=True)
    torch.cuda.nvtx.range_pop()
    
    torch.cuda.nvtx.range_push("2. DMA Transfer B to GPU")
    B_gpu = B_gpu.to("cuda", non_blocking=True)
    torch.cuda.nvtx.range_pop()


    torch.cuda.nvtx.range_push("3. GPU MatMul Operation")
    A_gpu = torch.matmul(A_gpu, A_gpu)
    torch.cuda.nvtx.range_pop()
    
    torch.cuda.nvtx.range_push("4. GPU Softmax Operation")
    B_gpu = torch.softmax(B_gpu, dim=0)
    torch.cuda.nvtx.range_pop()
    
    # Final sync to ensure all operations complete
    torch.cuda.synchronize()
    

    """
    EXTRA INFO on  : Freeing GPU memory
    When we exit this function, A_gpu and A_gpu would be garbage collected. 
    But with "nvidia-smi", you would still see Memory  usage.
    It's due to an optimization on pytorch side to quickly allocate
    GPU memory for potential future tensors. 
    To remove those cache, run : torch.cuda.empty_cache()
    """

def main():

    warmup_iters = 2
    print("Running warmup iterations...")
    for _ in range(warmup_iters):
        cuda_interaction_demo()
    
    if args.profile:
        print("Starting profiled iteration...")
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push("cuda_interaction_demo")
        cuda_interaction_demo()
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
        print("Profiling complete")

if __name__ == "__main__":
    main()
    # Track peak memory usage
    # mem = max(memory_usage(proc=main))
    # print(f"Peak host memory usage: {mem:.2f} MiB")