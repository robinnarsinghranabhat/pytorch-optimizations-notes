### How to to Overlap Communication and Computation in PyTorch 

Communication Mostly entails : Data transfers like :  CPU and GPU, GPU to GPU e.t.c within same machine or across different machines. 

Computation Mostly entails : Operations like matrix multiplication.

Since different hardwares exist for DMA takes care of first one, while computations happen done in GPU cores. 
Hence, we can overlap these two operations.

### Explaining the Profiler Output Below
In our example, we do an `all_reduce` operation i.e We create the same Tensor `comm_tensor` in GPU0 and GPU1 each. Then we sum them up and update `comm_tensor`. This would require, GPU0 to send it's tensor to GPU1 and vice-versa. 
We have two processes (pytorch programs), each working on separate GPUs. Visualization below is for one of the process (say process 0 and GPU0). We can see, in `all_reduce`, two data transfers take place for each process : 
- Data transfer from GPU0 Memory to System-Memory (reduce-scatter)  
- Again, Data transfer from System-Memory to GPU1 Memory (all-gather)

But, only thing of concern is not this implementation detail, but that fact that, while this data-transfer (communication) happens, 
other works are running parallely.

### Environment this program was tested on 
Single node with two gpus. These GPUS connected with PCIE Bus.

### Generating Profiling Results :  
`torchrun --nproc_per_node=2 overlap_comm_compute.py`

The output trace will be saved in the `./logs` directory, which you can load in your chromium browser at `chrome://tracing` 

### Possible Performance Enchancements :
Consider NUMA Affinity ( https://github.com/pytorch/pytorch/issues/115305 ). 


