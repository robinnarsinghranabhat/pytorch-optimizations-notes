### Minimal DDP training in pytorch 

### Environment
Single node with two gpus connected over PCIE , so not direct GPU-GPU data transfer option). 

### How to train 
- `python ddp.py` : This is our updated ddp implementation
- `python ddp_slower.py` : This is the naive ddp for comparison
- `python ddp_profiled.py` : This is the profiled version of our `ddp.py`


## Implemention Details
Back in high school, we used to joke about math teachers solving that first simple example, then assigning whole chapter homework.

I had covered the fundamentals on **[CUDA Streams](https://github.com/robinnarsinghranabhat/pytorch-optimizations-notes/tree/main/01.%20cuda-stream-tutorial)** and **[Communication-Computation Overlap](https://github.com/robinnarsinghranabhat/pytorch-optimizations-notes/tree/main/02.%20async-communication-tutorial-torch)**.

To build something more substantial ( while i still had an access to multi-gpu node), implemented a **minimal, pedagogical DDP that overlaps gradient communication during backpropagation**. I extend on top of [This](https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html) official pytorch article by Seb Arnold. Key Difference is : instead of averaging gradients across GPUs only after `loss.backward()` completes, we start communicating gradients as soon as they're computed for each layer.

### Results
With Updated version, got **median 1.5 second improvement per epoch**. This gave a feel for potential time effective communication it can save on those YOLO trainings they talk about.

I encourage to checking out the [async communication tutorial](https://github.com/robinnarsinghranabhat/pytorch-optimizations-notes/tree/main/02.%20async-communication-tutorial-torch) first, where I explain profiler visualizations of communication-computation overlap with a simple example.
