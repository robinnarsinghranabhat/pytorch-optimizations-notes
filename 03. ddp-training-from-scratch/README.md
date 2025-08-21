### Minimal DDP training in pytorch 

### Environment
Single node with two gpus connected over PCIE , so not direct GPU-GPU data transfer option). 

### How to train 
- `python ddp.py` : This is our updated ddp implementation
- `python ddp_slower.py` : This is the naive ddp
- `python ddp_profiled.py` : This is the profiled version of our `ddp.py`