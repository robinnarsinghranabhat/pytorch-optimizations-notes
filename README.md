# pytorch-cuda-kernel-optimization-example (Work in Progress)
Guidelines to program deep learning applications. A unified collection of scattered ideas focused on imporving performance of ML programs.

Check out `./streaming` folder. You can ignore rest.

## How to run : 
1. Compile : `python3 setup.py install`

## Insights
1. CPP forward kernel is faster than custom pytorch forward because : 
- During forward pass, Each Pytorch function invokes python interpreter, which further calls the underlying CPP api. Only there, CUDA kernels are executed. Finally, results flow back to python interpreter.
- In custom CPP operation, after interpreter calls the custom CPP forward function, the pytorch program only receives the final output. The intermediate calls like (torch::addmm, torch::sigmoid e.t.c) happen in C++ land.
