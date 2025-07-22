## Profiling with CUDA Streams
### https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
### Personal :: https://gemini.google.com/app/061daa91e4502050
import torch
import torch.profiler

def run_sequential(size):
    """Runs creation and matmul sequentially on the default stream."""
    A = torch.randn(size, size, device='cuda')
    B = torch.randn(size, size, device='cuda')
    C = torch.mm(A, A)
    D = torch.mm(B, B)
    torch.cuda.synchronize()

def run_with_streams(size):
    """Runs creation and matmul concurrently on two separate streams."""
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()

    with torch.cuda.stream(s1):
        A = torch.randn(size, size, device='cuda')
        C = torch.mm(A, A)

    with torch.cuda.stream(s2):
        B = torch.randn(size, size, device='cuda')
        D = torch.mm(B, B)

    torch.cuda.synchronize()

# --- Profiling Setup ---
size = 5000
log_dir = "./log"


# --- Profile the Sequential Case ---
print("Profiling sequential execution...")
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(f'{log_dir}/sequential'),
    record_shapes=True,
    with_stack=True
) as prof_sequential:
    for _ in range(4): # 1 wait, 1 warmup, 2 active
        run_sequential(size)
        prof_sequential.step() # Mark the end of an iteration

# --- Profile the Streams Case ---
print("Profiling execution with streams...")
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(f'{log_dir}/with_streams'),
    record_shapes=True,
    with_stack=True
) as prof_stream:
    for _ in range(4): # 1 wait, 1 warmup, 2 active
        run_with_streams(size)
        prof_stream.step() # Mark the end of an iteration

print(f"Profiling complete. Traces saved in '{log_dir}' directory.")