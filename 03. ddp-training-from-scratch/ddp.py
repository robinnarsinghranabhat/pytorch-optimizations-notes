"""
Last time, I talked about leveraging cuda streams to overlap independent `computations` and `data transfer` tasks.
Remember, pytorch operations like torch.matmul(A,B) are queued as tasks in a "Default CUDA stream".
All tasks in the same stream are executed in order, one after another; even if they are independent.

This a minimal Distributed-Data-Parallel training in pytorch. Main goal is to show 
how to use multiple streams to overlap computation and communication requests made to CUDA.

Basic Idea : During backward pass, after calculating gradients of Lth layer, an independent communication request is queued in a 
separate stream to average gradients of Lth layer across all GPUs. At the same time, in our default stream, we continue with
computing gradients for (L-1)th layer.

I also show how create a very simple bucketing strategy to collect gradients of multiple tensors before 
sending them for communication.
"""

import time
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

""" Distributed Asynchronous SGD Example """
def run(rank, size):
    torch.manual_seed(1234)
    device = torch.device("cuda:{}".format(rank))
    train_set, bsz = partition_dataset()

    model = Net().to(device)

    # Register a backward hook on each parameter
    handles = []
    for param in model.parameters():
        if param.requires_grad:  # Ensure it's a leaf tensor that requires gradients
            hook_handle = param.register_post_accumulate_grad_hook(communication_hook)
            handles.append(hook_handle)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set.dataset) / float(bsz))

    times = []
    for epoch in range(10):
        epoch_loss = 0.0
        start_time = time.time()
        for data, target in train_set:
            
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            data = data.view(data.size(0), -1)  # Flatten the input

            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            
            # Backward pass - this triggers the communication hooks
            loss.backward()

            # CRITICAL: Wait for all pending communication operations to complete
            # and update gradients before taking the optimizer step
            bucket.wait_for_pending_operations()
            
            # Optional: Ensure communication stream operations complete before optimizer step
            # This provides an extra safety net
            # torch.cuda.current_stream().wait_stream(comm_stream)

            # Now it's safe to update parameters
            optimizer.step()

            # Synchronize all processes before proceeding to next batch
            torch.distributed.barrier()

        if rank == 0:
            print('--- Rank', dist.get_rank(), ', epoch', epoch, ':', epoch_loss / num_batches)

        end_time = time.time()
        diff = end_time - start_time
        times.append(diff)
        if rank == 0:
            print('Rank', dist.get_rank(), ', epoch', epoch, ':', diff)

    if rank == 0:
        print('Rank', dist.get_rank(), ', total time:', sum(times))

    # Clean up: Remove hooks
    for handle in handles:
        handle.remove()


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
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

