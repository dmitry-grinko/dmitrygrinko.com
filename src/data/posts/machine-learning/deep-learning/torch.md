# PyTorch

PyTorch is Facebook's open-source deep learning framework that has gained massive popularity in the research community due to its dynamic computation graphs and intuitive Python-first design. It provides maximum flexibility and speed for deep learning research and production.

## Key Features

- **Dynamic computation graphs** - Define-by-run approach, easy to debug
- **Pythonic** - Natural Python integration, works with standard debuggers
- **Strong GPU acceleration** - Seamless CUDA integration
- **Research friendly** - Preferred by many researchers and in academic papers
- **Production ready** - TorchScript and ONNX export for deployment

## 1. Installation

```bash
# CPU only
pip install torch torchvision

# With CUDA (check pytorch.org for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

```python
import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## 2. Tensors

### 2.1 Creating Tensors

```python
import torch

# From data
t = torch.tensor([1.0, 2.0, 3.0])
matrix = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

# Factory functions
zeros = torch.zeros(3, 4)
ones = torch.ones(2, 3)
rand = torch.rand(5, 5)          # Uniform [0, 1)
randn = torch.randn(5, 5)        # Standard normal
arange = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]

# From NumPy (shares memory)
import numpy as np
arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)
```

### 2.2 Tensor Operations

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Element-wise
print(a + b)          # [5, 7, 9]
print(a * b)          # [4, 10, 18]

# Matrix multiplication
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = A @ B             # or torch.matmul(A, B)
print(C.shape)        # [3, 5]

# Reshaping
x = torch.randn(4, 6)
print(x.view(24).shape)       # [24]
print(x.reshape(2, 12).shape) # [2, 12]
print(x.permute(1, 0).shape)  # [6, 4]

# Move to GPU
x_gpu = x.to(device)
```

### 2.3 Autograd

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1

y.backward()
print(x.grad)  # dy/dx = 2x + 2 = 8.0

# Disable gradient tracking for inference
with torch.no_grad():
    z = x ** 2  # No gradient tracked

# Detach from computation graph
x_detached = x.detach()
```

## 3. Building Neural Networks

### 3.1 nn.Module

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        return self.net(x)

model = MLP(784, 256, 10).to(device)
print(model)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### 3.2 Common Layers

```python
# Linear
nn.Linear(in_features=128, out_features=64)

# Convolutional
nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)

# Pooling
nn.MaxPool2d(kernel_size=2, stride=2)
nn.AdaptiveAvgPool2d(output_size=(1, 1))

# Recurrent
nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
nn.GRU(input_size=64, hidden_size=128, batch_first=True)

# Normalization
nn.BatchNorm2d(num_features=32)
nn.LayerNorm(normalized_shape=64)

# Activation
nn.ReLU()
nn.GELU()
nn.Sigmoid()
nn.Softmax(dim=-1)
```

## 4. Training Loop

### 4.1 Dataset and DataLoader

```python
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# Built-in dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader  = DataLoader(test_dataset,  batch_size=256, shuffle=False)

# Custom dataset
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

### 4.2 Training and Evaluation

```python
import torch.optim as optim

model = MLP(784, 256, 10).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0

    for x, y in loader:
        x, y = x.view(x.size(0), -1).to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(x)
        correct += (logits.argmax(1) == y).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.view(x.size(0), -1).to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * len(x)
            correct += (logits.argmax(1) == y).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)


for epoch in range(20):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, test_loader, criterion)
    scheduler.step()
    print(f"Epoch {epoch+1:2d} | train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")
```

## 5. Convolutional Neural Network

```python
class CNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

## 6. Saving and Loading

```python
# Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': val_loss,
}, 'checkpoint.pt')

# Load checkpoint
checkpoint = torch.load('checkpoint.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Save model weights only
torch.save(model.state_dict(), 'weights.pt')
model.load_state_dict(torch.load('weights.pt', map_location=device))
```

## 7. TorchScript for Production

```python
# Trace (works for fixed control flow)
example_input = torch.randn(1, 784)
traced_model = torch.jit.trace(model, example_input)
traced_model.save('model_traced.pt')

# Script (handles dynamic control flow)
scripted_model = torch.jit.script(model)
scripted_model.save('model_scripted.pt')

# Load and run
loaded = torch.jit.load('model_scripted.pt')
output = loaded(example_input)
```

## 8. Conclusion

PyTorch's dynamic graph approach makes it feel like regular Python, which is why it dominates research. Key takeaways:

- **`nn.Module`** is the base class for all models — subclass it for custom architectures
- **`DataLoader`** handles batching, shuffling, and parallel data loading
- **`autograd`** tracks operations automatically — just call `.backward()`
- **`torch.no_grad()`** disables gradient tracking for inference (saves memory and time)
- **TorchScript** bridges the gap between research flexibility and production performance

PyTorch's ecosystem — including `torchvision`, `torchaudio`, `torchtext`, and Hugging Face Transformers — covers virtually every deep learning domain.
