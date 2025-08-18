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
        # NOTE : This is new copy in GPU memory.
        merged_gradient_tensors = bucket.collect_tensors(tensor.grad)

        if merged_gradient_tensors is not None:
            # Just ensuring that this tensor is not required to be tracked in autograd's computational graph"
            assert merged_gradient_tensors.requires_grad == False

            # We want CPU to queue the communication operation on a separate stream
            # How this communication happens is not important to us. Based on backends used and hardware supported, 
            # tensors could be communicated directly between GPUs (nvlink) or through the slower GPU-CPU-GPU path (PCIE).
            with torch.cuda.stream(comm_stream):
                
                handle = dist.all_reduce(merged_gradient_tensors, op=dist.ReduceOp.SUM, async_op=True)
                # Wait ensures the operation is enqueued, but not necessarily complete.
                handle.wait()
                merged_gradient_tensors /= WORLD_SIZE

                # Update the Pameters in the bucket with new averaged gradients
                # NOTE : Update below must also happen inside this same stream
                # Otherwise, we can't guarentee `merged_gradient_tensors` have been updated before
                # the operation below.
                bucket.update_tensor_grads(merged_gradient_tensors)

                # if dist.get_rank() == 0:
                #     print("Commuunicattion done for Tensor of size ", merged_gradient_tensors.numel())

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



""" Distributed Asynchronous SGD Example """
def run(rank, size):
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

    for epoch in range(10):
        epoch_loss = 0.0
        for data, target in train_set:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            data = data.view(data.size(0), -1)  # Flatten the input

            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward() # Gradients are calculated here

            # Wait for all operations in communication queue to complete before updating the parameters
            torch.cuda.current_stream().wait_stream(comm_stream)
            torch.distributed.barrier()  # Ensure all processes reach this point before proceeding 
            optimizer.step()

        print('Rank ', dist.get_rank(), ', epoch ',
              epoch, ': ', epoch_loss / num_batches)



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



### MY DDP RUN
# Rank  1 , epoch  0 :  2.2392996650631143
# Rank  0 , epoch  0 :  2.2396173760042353
# Rank  1 , epoch  1 :  1.9120400719723458
# Rank  0 , epoch  1 :  1.9134451575198417
# Rank  1 , epoch  2 :  1.1675578184046989
# Rank  0 , epoch  2 :  1.1716073498887531
# Rank  1 , epoch  3 :  0.7390157832937726
# Rank  0 , epoch  3 :  0.7463569843162925
# Rank  1 , epoch  4 :  0.5698894686618093
# Rank  0 , epoch  4 :  0.5788268734843044
# Rank  1 , epoch  5 :  0.4829946756362915
# Rank  0 , epoch  5 :  0.49047932988506254
# Rank  1 , epoch  6 :  0.42881107330322266
# Rank  0 , epoch  6 :  0.4380209380287235
# Rank  1 , epoch  7 :  0.3939585039171122
# Rank  0 , epoch  7 :  0.40175293411238716
# Rank  1 , epoch  8 :  0.3683412898395021
# Rank  0 , epoch  8 :  0.3754793713658543
# Rank  1 , epoch  9 :  0.3487344929727457
# Rank  0 , epoch  9 :  0.35489133303448306


### ORIGNAL DDP RUN
# Rank  0 , epoch  0 :  2.2396173760042353
# Rank  1 , epoch  0 :  2.2392996650631143
# Rank  0 , epoch  1 :  1.9134451575198417
# Rank  1 , epoch  1 :  1.9120400719723458
# Rank  0 , epoch  2 :  1.1716073498887531
# Rank  1 , epoch  2 :  1.1675578184046989
# Rank  0 , epoch  3 :  0.7463569843162925
# Rank  1 , epoch  3 :  0.7390157832937726
# Rank  0 , epoch  4 :  0.5788268734843044
# Rank  1 , epoch  4 :  0.5698894686618093
# Rank  0 , epoch  5 :  0.49047932988506254
# Rank  1 , epoch  5 :  0.4829946756362915
# Rank  0 , epoch  6 :  0.4380209380287235
# Rank  1 , epoch  6 :  0.42881107330322266
# Rank  0 , epoch  7 :  0.40175293411238716
# Rank  1 , epoch  7 :  0.3939585039171122
# Rank  0 , epoch  8 :  0.3754793713658543
# Rank  1 , epoch  8 :  0.3683412898395021
# Rank  0 , epoch  9 :  0.35489133303448306
# Rank  1 , epoch  9 :  0.3487344929727457