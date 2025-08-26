## Recipes on writing better Pytorch

Keeping GPU busy is essential for efficient training and inference, especially in a real-world distributed setting (Multiple Nodes, Each having Multiple GPUs).
But in distributed setting, if not careful, programs could be very inefficient as, GPUs could remain idle majority of time, waiting for data (could be intermediate activations, gradients e.t.c)


Topics covered : 
- Visualizing pytorch programs better with profiling tools (torch Profiler, Nvidia Nsight Systems) 
- How Program Execution happens, where CPU makes computation request to the GPU
- Utilize Concept of "CUDA Stream" to write efficient programs that can keep GPU Busy ( matmul on GPU, or some CPU operations even)
