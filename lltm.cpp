A_gpu = torch.zeros((8000, 8000), device="cpu", pin_memory=True)
B_gpu = torch.ones((8000, 8000), device="cpu", pin_memory=True)


A_gpu = A_gpu.to("cuda", non_blocking=True)
B_gpu = B_gpu.to("cuda", non_blocking=True)

A_gpu = torch.matmul(A_gpu, A_gpu)
B_gpu = torch.softmax(B_gpu, dim=0)
