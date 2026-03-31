import paddle
print(paddle.is_compiled_with_cuda())   # True if CUDA build
print(paddle.device.get_device())       # e.g. "gpu:0" or "cpu"