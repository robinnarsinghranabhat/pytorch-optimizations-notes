"""
This is a profiled version to DEBUG ddp.py:
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

# This is a separate CUDA stream for sending communication requests
comm_stream = torch.cuda.Stream()

class Bucket(object):
    """ 
    Naive Bucket implementation to hold tensors for communication.
    This is a simplified version of the bucket used in DDP.
    It holds upto a pre-defined (upto 2) tensors
    then uses it for communication when full.

    For our example, We have total 8 Parameter Tensors (4 layers, each with weight and bias).
    So, we collect gradients of 2 parameters (1 layer) in the bucket before sending them for communication.
    """

    TENSOR_LIMIT = 2
    def __init__(self, tensor_limit=TENSOR_LIMIT):
        self.tensors = []
        self.size = 0
        # Store pending communication handles and their associated data
        self.pending_operations = []
        
    def add(self, tensor):
        self.tensors.append(tensor)
        self.size += tensor.numel()
    
    def update_tensor_grads(self, merged_tensors, original_tensors):
        # Unflatten the buffer back into the original tensors
        utils.vector_to_parameters(merged_tensors, original_tensors)

    def clear(self):
        # Clear the bucket.
        self.tensors = []
        self.size = 0

    def collect_tensors(self, tensor):
        """ 
        Accumulate tensors in the bucket. If the bucket is full, 
        return the concatenated tensor to initiate the communication.
        """
        self.add(tensor)
        if len(self.tensors) < self.TENSOR_LIMIT:            
            return None, None
        else:
            # Concatenate all tensors in the bucket into a single tensor
            # NOTE : This creates a new Tensor i.e allocates additional GPU memory
            merged_tensors = utils.parameters_to_vector(self.tensors)
            # Return both the merged tensor and the original tensor list
            # original_tensors = self.tensors.copy()  # Keep a reference to original tensors
            return merged_tensors, self.tensors

    def wait_for_pending_operations(self):
        """
        Wait for all pending communication operations to complete and update gradients.
        This should only be called before optimizer.step() to ensure all gradients are properly averaged.
        """
        for handle, merged_tensor, original_tensors in self.pending_operations:
            # Wait for that `all_reduce` communication to finish 
            handle.wait()
            
            # Only Now, it's safe to average and update gradients
            merged_tensor /= WORLD_SIZE

            # We kept reference to the original tensors, so we can update their gradients
            # with the averaged gradients.
            self.update_tensor_grads(merged_tensor, original_tensors)
        
        # Clear pending operations
        self.pending_operations.clear()

    def add_pending_operation(self, handle, merged_tensor, original_tensors):
        """Add a pending communication operation to track."""
        self.pending_operations.append((handle, merged_tensor, original_tensors))


bucket = Bucket()


def communication_hook(tensor):
    # Collect tensors in the bucket
    merged_gradient_tensors, original_tensors = bucket.collect_tensors(tensor.grad)

    if merged_gradient_tensors is not None:
        # Just explicitly ensuring that that this tensor is not required to be tracked in 
        # the computational graph
        assert merged_gradient_tensors.requires_grad == False

        # We want CPU to queue the communication operation on a separate stream
        with torch.cuda.stream(comm_stream):
            # CPU invokes an asynchronous all_reduce operation and immediately continues to next line
            handle = dist.all_reduce(merged_gradient_tensors, op=dist.ReduceOp.SUM, async_op=True)
            
            # IMPORTANT: Don't process the tensor here! The all_reduce hasn't completed yet.
            # Instead, store the handle and tensor for later processing.
            bucket.add_pending_operation(handle, merged_gradient_tensors, original_tensors)

        # Clear the bucket for next batch of tensors
        bucket.clear()

    return tensor


class Net(nn.Module):
    """ Network architecture. """

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.drpout1 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(2048, 2048)
        self.drpout2 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(2048, 2048)
        self.drpout3 = nn.Dropout(0.3)
        self.fc5 = nn.Linear(2048, 2048)
        self.fc6 = nn.Linear(2048, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.drpout1(x)
        x = self.fc2(x)
        x = self.drpout2(x)
        x = self.fc3(x)
        x = self.drpout3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        return F.log_softmax(x, dim=1)

### For Profiling the Program ###
def trace_handler(p):

    print("\n\nExporting Chrome trace...")
    rank = dist.get_rank()
    f_name = f"./profiler_logs/rank_{rank}_" + str(p.step_num) + ".json"
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
     
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    