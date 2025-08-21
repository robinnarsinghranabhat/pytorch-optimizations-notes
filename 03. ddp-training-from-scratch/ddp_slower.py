"""
This implementation is based on Distributed Data Parallel (DDP) training example in PyTorch
where gradients are communicated after loss.backward() step (i.e after computing gradients)
https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html

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

""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size



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



""" Distributed Synchronous SGD Example """
def run(rank, size):
    torch.manual_seed(1234)
    device = torch.device("cuda:{}".format(rank))
    train_set, bsz = partition_dataset()

    
    model = Net().to(device)
    
    optimizer = optim.SGD(model.parameters(),
                          lr=0.01, momentum=0.5)
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
            loss.backward() # Gradients are calculated here

            average_gradients(model) #  gradient communcation happens here
            optimizer.step()
        
        if rank == 0:
            print('--- Rank ', dist.get_rank(), ', epoch ',
              epoch, ': ', epoch_loss / num_batches)


        end_time = time.time()
        diff = end_time - start_time
        times.append(diff)
        if rank == 0:
            print('Rank ', dist.get_rank(), ', epoch ', epoch, ': ', diff)
        
    if rank == 0:
        print('Rank ', dist.get_rank(), ', total time: ', sum(times))

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