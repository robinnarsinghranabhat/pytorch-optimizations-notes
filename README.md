# pytorch-cuda-kernel-optimization-example
Guidelines to Improve performance of custom pytorch modules

# Most tutorials feel like:
- Author actually has no good idea what's actually happening, navigated the docs and created tutorial (Practitioner/ Enthusiasts/ Learner)
- Guy is too smart and builds upon complex abstract concepts, which they assume the user knows ( Compiler Designers / Hardware guys )

# Target : 

## How to run : 
1. Compile : `python3 setup.py install`


## Insights
1. Just the CPP kernel is faster than custom pytorch forward because : 
- During forward pass, Each Pytorch function invoked python interpreter further calls the underlying CPU api and results flow back to python interpreter.
- In custom CPP operation, after initial call to CPP forward function, the interpreter only receives the final output. The intermediate calls like (torch::addmm, torch::sigmoid e.t.c) happen in C++ land.

2. Let's Try to Parallelize with CUDA Streams : 
https://stackoverflow.com/questions/11888772/when-to-call-cudadevicesynchronize
```
kernel1<<<X,Y>>>(...); // kernel start execution, CPU continues to next statement
kernel2<<<X,Y>>>(...); // kernel is placed in queue and will start after kernel1 finishes, CPU continues to next statement
cudaMemcpy(...); // CPU blocks until memory is copied, memory copy starts only after kernel2 finishes
```

https://stackoverflow.com/questions/52498690/how-to-use-cuda-stream-in-pytorch
```
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()
# Initialise cuda tensors here. E.g.:
# NOTE: These two instructions below happen sequentially by default as they are in same stream
A = torch.rand(1000, 1000, device = 'cuda')
B = torch.rand(1000, 1000, device = 'cuda') 
# Wait for the above tensors to initialise.
torch.cuda.synchronize()
with torch.cuda.stream(s1):
    C = torch.mm(A, A)
with torch.cuda.stream(s2):
    D = torch.mm(B, B)
# Wait for C and D to be computed.
torch.cuda.synchronize()
# Do stuff with C and D.

```

