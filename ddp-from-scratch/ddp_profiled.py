"""
Last time, I talked about levaraging cuda streams to overlap independent `computations` and `data transfer` tasks.
Remember, pytorch operations like torch.matmul(A,B) are queued as tasks in a "Default CUDA stream".
All tasks in the same stream are executed in order, one after another.

Here, implemented a minimal Distributed-Data-Parallel training in pytorch. Main goal is to show 
how to use multiple streams to overlap computation and communication requests made to CUDA.

Basic Idea : During backward pass, after calculating gradients of Lth layer, an independent communication request is queued in a 
separate stream to average gradients of Lth layer across all GPUs. At the same time, in our default stream, we continue with
computing gradients for (L-1)th layer.

I also show how create a very simple bucketing strategy to collect gradients of multiple tensors before 
sending them for communication.

PROFILING ADDED:
- Uses PyTorch profiler to capture CUDA events and timeline
- Records both computation and communication streams
- Exports Chrome trace for visualization
- Adds timing measurements for overlap analysis

"""
import torch
import torch.nn as nn
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.optim as optim
from math import ceil
import time

from ddp_utils import partition_dataset

WORLD_SIZE = 2

# Create a CUDA stream for sending communication requests
comm_stream = torch.cuda.Stream()

class Bucket(object):
    """ 
    A Bucket to hold tensors for communication.
    This is a simplified version of the bucket used in DDP.
    It holds upto a pre-defined (upto 2) number of tensors
    then uses it for communication when full.

    For our example, We have total 8 parameters (4 layers, each with 2 parameters - weight and bias).
    So, we collect gradients of 2 parameters (1 layer) in the bucket before sending them for communication.
    """

    TENSOR_LIMIT = 2 # Default :  only collect 2 tensors in the bucket
    def __init__(self, tensor_limit=TENSOR_LIMIT):
        self.tensors = []
        self.size = 0
        
    def add(self, tensor):
        self.tensors.append(tensor)
        self.size += tensor.numel()
    
    def update_tensor_grads(self, merged_tensors):
        # Unflatten the buffer back into the original tensors
        utils.vector_to_parameters(merged_tensors, self.tensors)

    def clear(self):
        # Clear the bucket.
        self.tensors = []
        self.size = 0

    def collect_tensors(self, tensor):
        """ 
        Accumulate tensors in the bucket. If the bucket is full, 
        return the concatenated tensor.
        """
        self.add(tensor)
        if len(self.tensors) < self.TENSOR_LIMIT:            
            return
        else:
            # Concatenate all tensors in the bucket into a single tensor
            # NOTE : This creates a new Tensor i.e allocates additional GPU memory
            merged_tensors =  utils.parameters_to_vector(self.tensors)
            return merged_tensors


bucket = Bucket()


def communication_hook(tensor):
    with torch.profiler.record_function("gradient_communication_hook"):
        # NOTE : This is new copy in GPU memory.
        merged_gradient_tensors = bucket.collect_tensors(tensor.grad)

        if merged_gradient_tensors is not None:
            # Just ensuring that this tensor is not required to be tracked in autograd's computational graph"
            assert merged_gradient_tensors.requires_grad == False

            # We want CPU to queue the communication operation on a separate stream
            # How this communication happens is not important to us. Based on backends used and hardware supported, 
            # tensors could be communicated directly between GPUs (nvlink) or through the slower GPU-CPU-GPU path (PCIE).
            with torch.cuda.stream(comm_stream):
                with torch.profiler.record_function("allreduce_operation"):
                    handle = dist.all_reduce(merged_gradient_tensors, op=dist.ReduceOp.SUM, async_op=True)
                    # Wait ensures the operation is enqueued, but not necessarily complete.
                    handle.wait()
                    merged_gradient_tensors /= WORLD_SIZE

                with torch.profiler.record_function("update_bucket_gradients"):
                    # Update the Pameters in the bucket with new averaged gradients
                    # NOTE : Update below must also happen inside this same stream
                    # Otherwise, we can't guarentee `merged_gradient_tensors` have been updated before
                    # the operation below.
                    bucket.update_tensor_grads(merged_gradient_tensors)

                    # if dist.get_rank() == 0:
                    #     print(f"Communication done for Tensor of size {merged_gradient_tensors.numel()}")

            bucket.clear()

        return tensor


class Net(nn.Module):
    """ Network architecture. """

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc4(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))))
        return F.log_softmax(x)


# for profiling
def trace_handler(p):
    # output = p.key_averages().table(sort_by=sort_by_keyword, row_limit=10)
    # print(output)
    print("\n\nExporting Chrome trace...")
    rank = dist.get_rank()
    f_name = f"./profiler_logs/rank_{rank}_" + str(p.step_num) + ".json"
    # only save if file does not exist
    if not os.path.exists(f_name):
        p.export_chrome_trace(f_name)
    else:
        print(f"File {f_name} already exists. Not overwriting.")


""" Distributed Asynchronous SGD Example """
def run(rank, size):

    PROFILE_EPOCHS = 5  # Number of epochs to profile
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()

    device = torch.device("cuda:{}".format(rank))
    model = Net().to(device)
    
    # Register a backward hook on each parameter
    handles = [] # Store handles to remove hooks later if needed
    for param in model.parameters():
        if param.requires_grad:  # Ensure it's a leaf tensor that requires gradients
            hook_handle = param.register_post_accumulate_grad_hook(communication_hook)
            handles.append(hook_handle)

    optimizer = optim.SGD(model.parameters(),
                          lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set.dataset) / float(bsz))

    # Initialize profiler
    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
        on_trace_ready=trace_handler,
        # torch.profiler.tensorboard_trace_handler(f'./profiler_logs/rank_{rank}'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )

    
    prof.start()
    # Loop runs for PROFILE_EPOCHS iterations
    for index, (data, target) in enumerate(train_set):
        # Only profile first few epochs to avoid huge trace files
        if index >= PROFILE_EPOCHS:
            break
        data, target = data.to(device), target.to(device)
        
        
        # Profile specific operations
        with torch.profiler.record_function("forward_pass"):
            optimizer.zero_grad()
            data = data.view(data.size(0), -1)  # Flatten the input
            output = model(data)
            loss = F.nll_loss(output, target)
            # epoch_loss += loss.item()

        with torch.profiler.record_function("backward_pass"):
            loss.backward() # Gradients are calculated here

        with torch.profiler.record_function("synchronization"):
            # Wait for all operations in communication queue to complete before updating the parameters
            torch.cuda.current_stream().wait_stream(comm_stream)
            torch.distributed.barrier()  # Ensure all processes reach this point before proceeding 

        with torch.profiler.record_function("optimizer_step"):
            optimizer.step()

        PROFILE_EPOCHS += 1
        # Step profiler
        prof.step()        
        
    prof.stop()

        
    # Export additional profiling data
    print(f"\nProfiling complete for rank {rank}!")
    print(f"Check ./profiler_logs/ for Chrome traces")

    # Print stream information
    print(f"\nRank {rank} Stream Information:")



def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    
    # Create profiler output directory
    os.makedirs('./profiler_logs', exist_ok=True)
    
    fn(rank, size)


if __name__ == "__main__":
    world_size = WORLD_SIZE
    processes = []
    if "google.colab" in sys.modules:
        print("Running in Google Colab")
        mp.get_context("spawn")
    else:
        mp.set_start_method("spawn")
    
    print("Starting DDP training with profiling...")
    print("Profiling outputs:")
    print("1. TensorBoard logs: ./profiler_logs/")
    print("2. Chrome traces: ./ddp_trace_rank_X.json")
    
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    