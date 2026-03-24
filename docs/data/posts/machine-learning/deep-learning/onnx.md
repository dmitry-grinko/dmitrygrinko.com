# ONNX

ONNX (Open Neural Network Exchange) is an open standard for representing machine learning models. It enables interoperability between different machine learning frameworks, allowing you to train a model in one framework and deploy it in another.

## Key Features

- **Framework interoperability** - Move models between PyTorch, TensorFlow, scikit-learn, and more
- **Standardized representation** - Common format for neural networks and classical ML models
- **Broad ecosystem support** - Supported by all major ML frameworks and hardware vendors
- **Optimization tools** - ONNX Runtime provides hardware-specific optimizations
- **Hardware flexibility** - Deploy on CPU, GPU, NPU, and edge devices

## 1. Installation

```bash
pip install onnx onnxruntime
# For GPU inference:
pip install onnxruntime-gpu
```

```python
import onnx
import onnxruntime as ort
print("ONNX version:", onnx.__version__)
print("ORT version:", ort.__version__)
```

## 2. Exporting Models to ONNX

### 2.1 From PyTorch

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleModel()
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 784)

torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input':  {0: 'batch_size'},  # Variable batch size
        'output': {0: 'batch_size'}
    },
    opset_version=17
)

print("Model exported to model.onnx")
```

### 2.2 From TensorFlow / Keras

```python
import tf2onnx
import tensorflow as tf

# Build and train a Keras model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Convert to ONNX
input_signature = [tf.TensorSpec([None, 784], tf.float32, name='input')]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature)

with open('keras_model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())
```

### 2.3 From scikit-learn

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Train a scikit-learn model
X, y = load_iris(return_X_y=True)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, y)

# Convert to ONNX
initial_type = [('float_input', FloatTensorType([None, 4]))]
onnx_model = convert_sklearn(clf, initial_types=initial_type)

with open('sklearn_model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())
```

## 3. Validating an ONNX Model

```python
import onnx

# Load and check the model
model = onnx.load('model.onnx')
onnx.checker.check_model(model)
print("Model is valid")

# Inspect the graph
print("Inputs:")
for inp in model.graph.input:
    print(f"  {inp.name}: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")

print("Outputs:")
for out in model.graph.output:
    print(f"  {out.name}: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")

print("Opset version:", model.opset_import[0].version)
```

## 4. Running Inference with ONNX Runtime

### 4.1 Basic Inference

```python
import onnxruntime as ort
import numpy as np

# Create inference session
session = ort.InferenceSession('model.onnx', providers=['CPUExecutionProvider'])

# Get input/output names
input_name  = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"Input:  {input_name} {session.get_inputs()[0].shape}")
print(f"Output: {output_name} {session.get_outputs()[0].shape}")

# Run inference
x = np.random.randn(5, 784).astype(np.float32)
outputs = session.run([output_name], {input_name: x})
predictions = outputs[0]
print("Predictions shape:", predictions.shape)
```

### 4.2 GPU Inference

```python
# Use CUDA execution provider
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2 GB
    }),
    'CPUExecutionProvider'  # Fallback
]

session = ort.InferenceSession('model.onnx', providers=providers)
print("Using providers:", session.get_providers())
```

### 4.3 Batch Inference

```python
def batch_inference(session, data, batch_size=64):
    """Run inference in batches."""
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    results = []

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size].astype(np.float32)
        output = session.run([output_name], {input_name: batch})[0]
        results.append(output)

    return np.concatenate(results, axis=0)

data = np.random.randn(1000, 784).astype(np.float32)
predictions = batch_inference(session, data)
print("All predictions shape:", predictions.shape)
```

## 5. Optimizing ONNX Models

### 5.1 ONNX Runtime Optimizations

```python
from onnxruntime.transformers import optimizer

# Optimize a transformer model
optimized_model = optimizer.optimize_model(
    'model.onnx',
    model_type='bert',
    num_heads=12,
    hidden_size=768
)
optimized_model.save_model_to_file('model_optimized.onnx')
```

### 5.2 Quantization

Reduce model size and speed up inference by converting float32 weights to int8.

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

# Dynamic quantization (no calibration data needed)
quantize_dynamic(
    model_input='model.onnx',
    model_output='model_quantized.onnx',
    weight_type=QuantType.QInt8
)

# Compare sizes
import os
original_size  = os.path.getsize('model.onnx') / 1024
quantized_size = os.path.getsize('model_quantized.onnx') / 1024
print(f"Original:  {original_size:.1f} KB")
print(f"Quantized: {quantized_size:.1f} KB")
print(f"Reduction: {(1 - quantized_size/original_size)*100:.1f}%")
```

## 6. Verifying Correctness

Always verify that the ONNX model produces the same outputs as the original.

```python
import torch
import numpy as np

# Original PyTorch model
model.eval()
x_np = np.random.randn(3, 784).astype(np.float32)
x_torch = torch.from_numpy(x_np)

with torch.no_grad():
    torch_output = model(x_torch).numpy()

# ONNX Runtime
session = ort.InferenceSession('model.onnx')
ort_output = session.run(None, {'input': x_np})[0]

# Compare
max_diff = np.max(np.abs(torch_output - ort_output))
print(f"Max absolute difference: {max_diff:.2e}")
assert max_diff < 1e-5, "Outputs differ too much!"
print("Outputs match — export verified.")
```

## 7. Conclusion

ONNX bridges the gap between ML frameworks and deployment targets. Key takeaways:

- **Export once, run anywhere** — train in PyTorch, deploy with ONNX Runtime on any hardware
- **ONNX Runtime** is often faster than native framework inference due to graph-level optimizations
- **Quantization** can cut model size by 4x and speed up inference with minimal accuracy loss
- **Always verify** outputs after export — numerical differences can creep in
- **Dynamic axes** let you handle variable batch sizes and sequence lengths at inference time

ONNX is the standard choice when you need to deploy ML models across different environments, languages, or hardware platforms.
