# Keras

Keras is a high-level neural networks API that makes deep learning accessible to everyone. Originally developed as an independent library, Keras is now integrated into TensorFlow as `tf.keras`, providing an intuitive interface for building and training deep learning models.

## Key Features

- **User-friendly** - Simple, consistent interface optimized for fast experimentation
- **Modular** - Easy to configure neural network building blocks
- **Extensible** - Easy to add new modules and functionality
- **Python native** - No separate config files needed
- **Multi-backend** - Keras 3 supports TensorFlow, JAX, and PyTorch backends

## 1. Installation

```bash
pip install tensorflow  # includes tf.keras
# or for standalone Keras 3:
pip install keras
```

```python
import keras
print(keras.__version__)
```

## 2. Core Building Blocks

### 2.1 Layers

Layers are the fundamental building blocks of Keras models.

```python
import keras
import numpy as np

# Dense (fully connected) layer
dense = keras.layers.Dense(64, activation='relu')

# Apply to input
x = np.random.randn(10, 32).astype('float32')
output = dense(x)
print("Output shape:", output.shape)  # (10, 64)
```

### 2.2 Common Layer Types

```python
# Dense layers
keras.layers.Dense(units=128, activation='relu')

# Convolutional layers
keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')
keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')

# Pooling
keras.layers.MaxPooling2D(pool_size=(2, 2))
keras.layers.GlobalAveragePooling2D()

# Recurrent
keras.layers.LSTM(units=64, return_sequences=True)
keras.layers.GRU(units=64)

# Normalization and regularization
keras.layers.BatchNormalization()
keras.layers.Dropout(rate=0.5)
keras.layers.LayerNormalization()

# Reshaping
keras.layers.Flatten()
keras.layers.Reshape(target_shape=(7, 7, 64))
```

## 3. Model Building Approaches

### 3.1 Sequential API

Best for simple, linear stacks of layers.

```python
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation='softmax')
])

model.summary()
```

### 3.2 Functional API

For models with multiple inputs/outputs, skip connections, or shared layers.

```python
# Multi-input model
input_a = keras.Input(shape=(64,), name='input_a')
input_b = keras.Input(shape=(32,), name='input_b')

x_a = keras.layers.Dense(32, activation='relu')(input_a)
x_b = keras.layers.Dense(32, activation='relu')(input_b)

merged = keras.layers.Concatenate()([x_a, x_b])
output = keras.layers.Dense(1, activation='sigmoid')(merged)

model = keras.Model(inputs=[input_a, input_b], outputs=output)
```

### 3.3 Model Subclassing

For maximum flexibility and custom forward pass logic.

```python
class ResidualBlock(keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.dense1 = keras.layers.Dense(units, activation='relu')
        self.dense2 = keras.layers.Dense(units)
        self.bn = keras.layers.BatchNormalization()

    def call(self, x, training=False):
        residual = x
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.bn(x, training=training)
        return keras.activations.relu(x + residual)  # Skip connection


class ResNet(keras.Model):
    def __init__(self, n_classes):
        super().__init__()
        self.input_proj = keras.layers.Dense(64)
        self.blocks = [ResidualBlock(64) for _ in range(3)]
        self.classifier = keras.layers.Dense(n_classes, activation='softmax')

    def call(self, x, training=False):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, training=training)
        return self.classifier(x)
```

## 4. Compiling and Training

### 4.1 Compile

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)
```

### 4.2 Fit

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test  = x_test.reshape(-1, 784).astype('float32') / 255.0

history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=256,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]
)
```

### 4.3 Evaluate and Predict

```python
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")

predictions = model.predict(x_test[:5])
print("Predicted classes:", predictions.argmax(axis=1))
```

## 5. Optimizers

```python
# SGD with momentum
keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# Adam variants
keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01)

# RMSprop
keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)

# Learning rate schedules
lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.01,
    decay_steps=1000
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
```

## 6. Loss Functions

```python
# Classification
keras.losses.SparseCategoricalCrossentropy()   # integer labels
keras.losses.CategoricalCrossentropy()          # one-hot labels
keras.losses.BinaryCrossentropy()               # binary classification

# Regression
keras.losses.MeanSquaredError()
keras.losses.MeanAbsoluteError()
keras.losses.Huber(delta=1.0)                   # robust to outliers

# Custom loss
def focal_loss(y_true, y_pred, gamma=2.0):
    ce = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    p_t = keras.ops.exp(-ce)
    return ((1 - p_t) ** gamma) * ce
```

## 7. Callbacks

```python
callbacks = [
    # Save best checkpoint
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    # Stop early
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    # Reduce LR on plateau
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    ),
    # TensorBoard
    keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1
    )
]
```

## 8. Custom Layers and Metrics

### 8.1 Custom Layer

```python
class ScaledDotAttention(keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.scale = d_model ** -0.5
        self.softmax = keras.layers.Softmax(axis=-1)

    def call(self, query, key, value):
        scores = keras.ops.matmul(query, keras.ops.transpose(key, axes=[0, 2, 1]))
        scores = scores * self.scale
        weights = self.softmax(scores)
        return keras.ops.matmul(weights, value)
```

### 8.2 Custom Metric

```python
class F1Score(keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = keras.metrics.Precision()
        self.recall = keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * (p * r) / (p + r + keras.backend.epsilon())

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()
```

## 9. Saving and Loading

```python
# Save full model
model.save('model.keras')
model = keras.models.load_model('model.keras')

# Save weights only
model.save_weights('weights.weights.h5')
model.load_weights('weights.weights.h5')

# Export for inference (no training state)
model.export('inference_model')
```

## 10. Conclusion

Keras makes deep learning approachable without sacrificing flexibility. Key takeaways:

- **Sequential API** is great for simple linear stacks
- **Functional API** handles complex topologies cleanly
- **Subclassing** gives you full Python control for research
- **Callbacks** make training monitoring and control easy
- **Custom layers and losses** let you extend Keras for any use case

Whether you're prototyping a quick experiment or building a production model, Keras provides the right level of abstraction for the job.
