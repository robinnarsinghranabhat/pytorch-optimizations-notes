## Click here :: [Main Article](https://github.com/robinnarsinghranabhat/pytorch-optimizations-notes/tree/main/streaming/article.md)

## Prerequities 
1. `Nvidia Nsight Systems` profiler
2. A CUDA device


## How to generate files for visualizations
```
nsys profile --force-overwrite true  -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu --capture-range=cudaProfilerApi --cudabacktrace=true -x true -o pytorch_sequential python sequential_data_processing.py

nsys profile --force-overwrite true  -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu --capture-range=cudaProfilerApi --cudabacktrace=true -x true -o pytorch_streaming python streaming_data_processing.py

nsys profile --force-overwrite true  -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu --capture-range=cudaProfilerApi --cudabacktrace=true -x true -o minimal_demo python minimnal_profile_example.py
```
