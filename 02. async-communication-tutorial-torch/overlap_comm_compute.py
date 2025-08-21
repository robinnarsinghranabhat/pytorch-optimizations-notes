import torch
import torch.distributed as dist
import os
import time

def setup_distributed():
    """Initializes the distributed process group."""
    # torchrun will set these environment variables
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Initialize the process group with the NCCL backend
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set the device for the current process
    torch.cuda.set_device(local_rank)
    
    print(f"Initialized process rank {rank} of {world_size} on device cuda:{local_rank}")
    return rank, local_rank, world_size

def cleanup_distributed():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

def main():
    rank, local_rank, world_size = setup_distributed()
    device = f"cuda:{local_rank}"

    # A tensor for communication (e.g., gradients)
    comm_tensor = torch.ones(2000, 2000, device=device) * (rank + 1)
    
    # Tensors for computation (e.g., a model layer)
    A_cpu = torch.randn(2000, 2000, device="cpu", pin_memory=True)
    B_cpu = torch.randn(2000, 2000, device="cpu", pin_memory=True)
    # Create empty tensor to hold the result of computation in GPU memory
    C = torch.empty(2000, 2000, device=device)

    # --- 2. Create CUDA Stream for communication ---
    comm_stream = torch.cuda.Stream(device=device)

    # Warm-up iterations to avoid one-off startup costs in the profile
    print("Running warm-up iterations...")
    for _ in range(2):
        handle = dist.all_reduce(comm_tensor, op=dist.ReduceOp.SUM, async_op=True)
        A = A_cpu.to(device, non_blocking=True)
        B = B_cpu.to(device, non_blocking=True)
        C = torch.matmul(A, B)
        handle.wait()
        # Now adding is safe. Without handle.wait(), we could add before comm_tensor was updated.
        C += comm_tensor
    
    # Synchronize before starting the profiler to get a clean trace
    torch.cuda.synchronize()

    print("Starting profiling...")
    
    # --- 3. Profile the Overlapped Operations ---
    # The output trace will be saved in the `./logs` directory
    trace_dir = "./profiler_logs"
    os.makedirs(trace_dir, exist_ok=True)
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=2, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
        record_shapes=True,
        with_stack=True,
        profile_memory=True
    ) as prof:
        for step in range(2): # The schedule above will only record 3 steps
            # Ensure the computation and communication are on separate streams
            
            # --- Start Communication on comm_stream ---
            with torch.cuda.stream(comm_stream):
                with torch.profiler.record_function(f"COMMUNICATION_STEP_{step}"):
                    # This call is non-blocking. It returns a handle immediately.
                    work_handle = dist.all_reduce(comm_tensor, op=dist.ReduceOp.SUM, async_op=True)
                
            # --- Start Computation on default stream ---
            # This happens immediately after the all_reduce is *enqueued*,
            # without waiting for it to complete.
            with torch.profiler.record_function(f"A_MEMORY_TRANSFER_STEP_{step}"):
                A = A_cpu.to(device, non_blocking=True)

            with torch.profiler.record_function(f"B_MEMORY_TRANSFER_STEP_{step}"):
                B = B_cpu.to(device, non_blocking=True)
            
            with torch.profiler.record_function(f"MATRIX_MULTIPLY_STEP_{step}"):
                C = torch.matmul(A, B) + torch.matmul(A, B) + torch.matmul(A, B)
            # --- Wait for comm_tensor to be updated  before proceeding --
            # As we have launched some tasks, we can wait for the communication to complete. 
            # computation will also happen in parallel
            with torch.profiler.record_function(f"WAIT_FOR_QUEUEING_{step}"):
                work_handle.wait()
            

            # Now operation below is Safe 
            with torch.profiler.record_function(f"SAFE_MATMUL_{step}"):
                C = torch.matmul(C, comm_tensor)
            prof.step() # Signal the profiler that one step is complete


    torch.cuda.synchronize()  
    prof.stop() 
    cleanup_distributed()
    print("Profiling finished. Trace saved to ./profiler_logs")

if __name__ == "__main__":
    main()
    # torchrun 