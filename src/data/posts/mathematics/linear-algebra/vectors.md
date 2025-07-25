# Vectors

Vectors are one of the foundational concepts in linear algebra and are essential for understanding modern machine learning. This article provides a deep dive into what vectors are, how they work, and how they're applied in data science and ML.

## 1. What Is a Vector?

A vector is an ordered list of numbers representing magnitude and direction in space. Each value in the list is called a **component**. Vectors can represent quantities such as position, velocity, and forces in physics, or features in machine learning.

### Examples:

- `[3]` → 1D vector
- `[3, 4]` → 2D vector
- `[1, -2, 5]` → 3D vector
- `[x₁, x₂, ..., xₙ]` → n-dimensional vector

## 2. Vector Notation and Types

- **Column vector**: A vertical list of values, often used in math.
- **Row vector**: A horizontal list, often used in programming.

In ML, vectors are often column vectors by convention, representing input features or weights.

## 3. Geometric Interpretation

Vectors can be visualized as arrows pointing from the origin to a point in space. For example, `[3, 4]` represents a point 3 units along x and 4 along y. The **length** of this vector is √(3² + 4²) = 5.

## 4. Vector Operations

### 4.1 Addition

Adding two vectors combines their components:

`[1, 2] + [3, 4] = [4, 6]`

### 4.2 Scalar Multiplication

Multiplying a vector by a number scales its length:

`3 * [2, -1] = [6, -3]`

### 4.3 Subtraction

`[5, 2] - [3, 1] = [2, 1]`

### 4.4 Dot Product (Inner Product)

The dot product returns a scalar:

`[1, 2] • [3, 4] = 1*3 + 2*4 = 11`

It's used to measure similarity and projection.

### 4.5 Norm (Length)

The length (L2 norm) of a vector `v = [x, y]` is √(x² + y²)

### 4.6 Unit Vector

A unit vector has length 1. Normalize a vector `v`:

`v_unit = v / ||v||`

## 5. Vectors in Machine Learning

### 5.1 Features as Vectors

Each data point is a vector of features. For example:

A house: `[size, bedrooms, age] = [1200, 3, 15]`

### 5.2 Weights as Vectors

In linear models (like logistic regression):

`y = w • x + b`

Where:

- `x` = input vector
- `w` = weight vector
- `b` = bias scalar

### 5.3 Gradient Descent

The gradient is a vector of partial derivatives. It shows the direction of steepest ascent. Gradient descent updates weights by moving in the opposite direction:

`w = w - α * ∇L(w)`

Where:

- `α` is learning rate
- `∇L(w)` is gradient of loss

### 5.4 Embeddings

In NLP and recommendation systems, words or users are represented as vectors in high-dimensional space. Similar vectors imply semantic or behavioral similarity.

### 5.5 Vectorized Operations

Using libraries like NumPy, operations are applied across vectors efficiently:

```python
import numpy as np
x = np.array([1, 2, 3])
w = np.array([0.5, 0.1, -0.2])
prediction = np.dot(x, w)
```

## 6. Vector Spaces

A vector space is a set of vectors that can be scaled and added while staying within the set. ML often works in such spaces (e.g., feature space).

Key properties:

- Closure under addition and scalar multiplication
- Contains a zero vector
- Each vector has an additive inverse

## 7. Projection and Similarity

### 7.1 Projection

Project one vector onto another to find its component in that direction:

`projₐb = (a • b / ||b||²) * b`

### 7.2 Cosine Similarity

Measures angle-based similarity:

`cos(θ) = (a • b) / (||a|| * ||b||)`

Used in recommender systems and NLP.

## 8. Vector Norms

- **L1 norm**: sum of absolute values → `|x₁| + |x₂| + ... + |xₙ|`
- **L2 norm**: Euclidean length → `√(x₁² + x₂² + ... + xₙ²)`
- **L∞ norm**: max absolute component

Used in different regularization techniques (L1 = Lasso, L2 = Ridge).

## 9. Conclusion

Vectors are everywhere in machine learning—from the data itself to the weights, gradients, and even model embeddings. A strong grasp of vector operations and their geometry is essential for building and understanding ML models.

Next time you see a model training, remember: behind every transformation, there's a vector being multiplied, normalized, projected, or compared.

