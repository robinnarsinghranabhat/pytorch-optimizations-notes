import torch
import psutil
import os
from memory_profiler import memory_usage
import argparse

def get_memory_usage():
    """Get current process memory usage in MB"""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

def test_memory_transfer(pin_in_advance):
    """Test memory usage for pinned memory allocation strategies"""
    size = (1024, 1024, 128)  # ~512MB tensor
    # size = (1)  # ~ 4 Byte tensor | Uncomment to check max-mem usage for baseline. 
    
    # Allocate tensor with or without pinned memory
    mem_before_tensor_allocation = get_memory_usage()
    x = torch.randn(size, device="cpu", pin_memory=pin_in_advance, dtype=torch.float32)
    curr_mem_alloc = get_memory_usage()
    # Pin memory later if not done during allocation
    if not pin_in_advance:
        x = x.pin_memory()
    
    # Transfer to GPU
    torch.cuda.synchronize()
    y = x.to("cuda", non_blocking=True)
    torch.cuda.synchronize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pin_in_advance', action='store_true', 
                       help='Allocate tensor with pinned memory vs pin later')
    args = parser.parse_args()
    
    initial_memory = get_memory_usage()
    peak_memory = max(memory_usage(proc=lambda: test_memory_transfer(args.pin_in_advance)))
    memory_overhead = peak_memory - initial_memory
    
    strategy = "Pin in advance" if args.pin_in_advance else "Pin later"
    print(f"{strategy}: Peak memory overhead = {memory_overhead:.1f} MB")