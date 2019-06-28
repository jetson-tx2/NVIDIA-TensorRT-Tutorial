import numpy as np
import onnx
import torch
import caffe2.python.onnx.backend as onnx_caffe2_backend

# This model is actually converted from a PyTorch one, and we are running caffe2 backend for inference
# We currently are trying to convert .onnx files to .trt ones with onnx2trt https://github.com/onnx/onnx-tensorrt
model = onnx.load('./artifacts/model4.onnx')

prepared_backend = onnx_caffe2_backend.prepare(model, device='CUDA:0')

out = prepared_backend.run(np.random.randn(1, 3, 128, 64).astype(np.float16))
