# TensorFlow

TensorFlow is Google's open-source platform for machine learning and deep learning. It provides a comprehensive ecosystem of tools, libraries, and community resources that enables researchers and developers to build and deploy ML-powered applications.

## Key Features

- **Flexible architecture** - Deploy computation to one or more CPUs or GPUs
- **Comprehensive ecosystem** - Complete platform from research to production
- **Multiple levels of abstraction** - From low-level operations to high-level APIs
- **Strong community** - Extensive documentation and community support
- **Production ready** - Used by Google and many other companies in production

## 1. Installation and Setup

```python
pip install tensorflow
```

```python
import tensorflow as tf
print(tf.__version__)

# Check GPU availability
print("GPUs available:", tf.config.list_physical_devices('GPU'))
```

## 2. Core Concepts

### 2.1 Tensors

Tensors are the fundamental data structure in TensorFlow — multi-dimensional arrays with a uniform type.

```python
import tensorflow as tf

# Scalar (rank 0)
scalar = tf.constant(3.14)

# Vector (rank 1)
vector = tf.constant([1.0, 2.0, 3.0])

# Matrix (rank 2)
matrix = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)

print("Scalar shape:", scalar.shape)
print("Vector shape:", vector.shape)
print("Matrix shape:", matrix.shape)
```

### 2.2 Variables

Variables hold mutable state — used for model weights.

```python
w = tf.Variable(tf.random.normal([3, 3]))
b = tf.Variable(tf.zeros([3]))

# Update in place
w.assign_add(tf.random.normal([3, 3]) * 0.01)
```

### 2.3 Automatic Differentiation

```python
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2 + 2 * x + 1  # y = x^2 + 2x + 1

# dy/dx = 2x + 2 = 8 at x=3
dy_dx = tape.gradient(y, x)
print("dy/dx at x=3:", dy_dx.numpy())  # 8.0
```


## 3. Building Models with Keras

TensorFlow's high-level API is `tf.keras`, which makes building neural networks straightforward.

### 3.1 Sequential API

```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

model.summary()
```

### 3.2 Functional API

For more complex architectures with multiple inputs/outputs or shared layers.

```python
inputs = keras.Input(shape=(784,))
x = keras.layers.Dense(128, activation='relu')(inputs)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(64, activation='relu')(x)
outputs = keras.layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```

### 3.3 Subclassing API

Full control by subclassing `keras.Model`.

```python
class MyModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(128, activation='relu')
        self.dropout = keras.layers.Dropout(0.2)
        self.dense2 = keras.layers.Dense(10, activation='softmax')

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.dense2(x)

model = MyModel()
```

## 4. Training a Model

### 4.1 Compile and Fit

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Load MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test  = x_test.reshape(-1, 784).astype('float32') / 255.0

history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1
)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
```

### 4.2 Custom Training Loop

```python
optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss_fn = keras.losses.SparseCategoricalCrossentropy()

@tf.function  # Compile to graph for speed
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        predictions = model(x_batch, training=True)
        loss = loss_fn(y_batch, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(128)

for epoch in range(10):
    epoch_loss = 0
    for step, (x_batch, y_batch) in enumerate(dataset):
        loss = train_step(x_batch, y_batch)
        epoch_loss += loss
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / (step + 1):.4f}")
```

## 5. Callbacks

```python
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'best_model.keras', monitor='val_accuracy', save_best_only=True
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=2
    ),
    keras.callbacks.TensorBoard(log_dir='./logs')
]

model.fit(x_train, y_train, epochs=50, callbacks=callbacks, validation_split=0.1)
```

## 6. Convolutional Neural Networks

```python
cnn_model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

x_train_cnn = x_train.reshape(-1, 28, 28, 1)
x_test_cnn  = x_test.reshape(-1, 28, 28, 1)

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(x_train_cnn, y_train, epochs=5, batch_size=64, validation_split=0.1)
```

## 7. Transfer Learning

```python
base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(128, activation='relu')(x)
outputs = keras.layers.Dense(5, activation='softmax')(x)

transfer_model = keras.Model(inputs, outputs)
transfer_model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

## 8. Saving and Loading Models

```python
# Save entire model
model.save('my_model.keras')
loaded_model = keras.models.load_model('my_model.keras')

# Weights only
model.save_weights('my_weights.weights.h5')
model.load_weights('my_weights.weights.h5')

# SavedModel format (for TensorFlow Serving)
model.export('saved_model_dir')
```

## 9. tf.data Pipeline

```python
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

dataset = (dataset
    .shuffle(buffer_size=10000)
    .batch(128)
    .prefetch(tf.data.AUTOTUNE)
)
```

## 10. Conclusion

TensorFlow is a complete ML platform covering everything from research to production. Key takeaways:

- **`tf.keras`** is the recommended high-level API for building models
- **`tf.GradientTape`** gives you full control over gradient computation
- **`tf.data`** pipelines make data loading efficient and scalable
- **`@tf.function`** compiles Python functions to TensorFlow graphs for speed
- **Transfer learning** lets you leverage pretrained models with minimal data

TensorFlow's ecosystem — including TensorBoard for visualization, TensorFlow Lite for mobile, and TensorFlow Serving for production — makes it a strong choice for end-to-end ML workflows.
