import torch
from memory_profiler import memory_usage
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pin-memory', action='store_true', default=False, help='Use pinned memory')
args = parser.parse_args()

def test_cuda_transfer():
    size = (1024, 1024, 64)  # ~128MB tensor
    # size = (1,1) # Uncomment this to check default memory use
    x = torch.randn(size, device="cpu")
    if args.pin_memory:
        x = x.pin_memory()
    y = x.to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    del y
    torch.cuda.empty_cache()

mem = max(memory_usage(proc=test_cuda_transfer))
print(f"Pin memory: {args.pin_memory} | Max host memory used: {mem:.2f} MiB")