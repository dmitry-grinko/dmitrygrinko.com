# Matrices and Matrix Operations

Matrices are the powerhouse data structure of machine learning. Every dataset, neural network weight, and transformation in ML can be represented as matrices. Understanding matrix operations is essential for grasping how algorithms work under the hood and why certain optimizations are possible.

## 1. What Are Matrices?

A matrix is a rectangular array of numbers arranged in rows and columns. Think of it as a collection of vectors stacked together.

### Examples:
```
A = [1  2  3]    (2×3 matrix: 2 rows, 3 columns)
    [4  5  6]

B = [7 ]         (3×1 matrix: column vector)
    [8 ]
    [9 ]

C = [1  0]       (2×2 matrix: square matrix)
    [0  1]
```

### Matrix Notation:
- `A[i,j]` or `A_{ij}` = element in row i, column j
- `A_{m×n}` = matrix with m rows and n columns

## 2. Types of Matrices in ML

### 2.1 Data Matrix
Most common in ML - each row is a sample, each column is a feature:

```
X = [feature1  feature2  feature3]  ← sample 1
    [   2.1      0.5      1.8  ]  ← sample 2
    [   1.9      0.3      2.1  ]  ← sample 3
    [   2.5      0.7      1.5  ]  ← sample 4
```

### 2.2 Weight Matrix
Neural network parameters:
```python
# Hidden layer weights: input_size × hidden_size
W1 = np.random.randn(784, 128)  # MNIST input to hidden

# Output layer weights: hidden_size × output_size  
W2 = np.random.randn(128, 10)   # Hidden to 10 classes
```

### 2.3 Identity Matrix
The "1" of matrix multiplication:
```
I = [1  0  0]
    [0  1  0]
    [0  0  1]
```

### 2.4 Zero Matrix
The "0" of matrix addition:
```
0 = [0  0  0]
    [0  0  0]
```

## 3. Basic Matrix Operations

### 3.1 Addition and Subtraction
Matrices must have the same dimensions:

```python
A = [[1, 2],     B = [[5, 6],
     [3, 4]]          [7, 8]]

A + B = [[6,  8],
         [10, 12]]

A - B = [[-4, -4],
         [-4, -4]]
```

### 3.2 Scalar Multiplication
Multiply every element by a scalar:

```python
A = [[1, 2],     2*A = [[2, 4],
     [3, 4]]            [6, 8]]
```

### 3.3 Element-wise Multiplication (Hadamard Product)
```python
A = [[1, 2],     B = [[5, 6],     A ⊙ B = [[5,  12],
     [3, 4]]          [7, 8]]              [21, 32]]
```

## 4. Matrix Multiplication

The most important operation in ML!

### 4.1 Definition
For `A_{m×k}` and `B_{k×n}`, the result `C_{m×n}` where:

`C[i,j] = Σ(A[i,k] × B[k,j])` for all k

### 4.2 Example
```
A = [1  2]    B = [5  6]    AB = [1×5+2×7  1×6+2×8] = [19  22]
    [3  4]        [7  8]         [3×5+4×7  3×6+4×8]   [43  50]
```

### 4.3 Practical Implementation
```python
def matrix_multiply(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    
    if cols_A != rows_B:
        raise ValueError("Cannot multiply: incompatible dimensions")
    
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    
    return result

# NumPy makes this much easier:
import numpy as np
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A @ B  # or np.dot(A, B)
```

## 5. Matrix Multiplication in Machine Learning

### 5.1 Linear Transformations
Every linear layer in a neural network is matrix multiplication:

```python
# Input: 784 features, Batch size: 32
X = np.random.randn(32, 784)  # 32 samples × 784 features

# Hidden layer: 784 → 128 neurons
W1 = np.random.randn(784, 128)
b1 = np.random.randn(128)

# Forward pass
hidden = X @ W1 + b1  # Shape: (32, 128)
```

### 5.2 Batch Processing
Matrix multiplication enables efficient batch processing:

```python
# Process one sample at a time (slow)
for sample in dataset:
    prediction = model(sample)

# Process entire batch at once (fast)
predictions = model(entire_batch)  # Vectorized!
```

### 5.3 Linear Regression
The normal equation uses matrix operations:

```python
def linear_regression(X, y):
    # θ = (X^T X)^(-1) X^T y
    XtX = X.T @ X
    Xty = X.T @ y
    theta = np.linalg.solve(XtX, Xty)
    return theta
```

## 6. Special Matrix Operations

### 6.1 Transpose
Flip rows and columns:

```python
A = [[1, 2, 3],     A^T = [[1, 4],
     [4, 5, 6]]            [2, 5],
                           [3, 6]]

# NumPy
A_transpose = A.T
# or A.transpose()
```

**Uses in ML:**
- Converting row vectors to column vectors
- Computing gradients: `∇W = X^T @ error`
- Covariance matrices: `Cov = (X - μ)^T @ (X - μ)`

### 6.2 Matrix Inverse
For square matrix A, A⁻¹ satisfies: `A @ A⁻¹ = I`

```python
A = np.array([[1, 2], [3, 4]])
A_inv = np.linalg.inv(A)
print(A @ A_inv)  # Should be close to identity matrix
```

**Warning:** Not all matrices have inverses! (singular matrices)

### 6.3 Determinant
A scalar value that indicates if a matrix is invertible:

```python
det_A = np.linalg.det(A)
# If det(A) = 0, matrix is not invertible
```

## 7. Advanced Matrix Concepts for ML

### 7.1 Rank
The dimension of the vector space spanned by the matrix columns:

```python
rank = np.linalg.matrix_rank(A)
```

**ML significance:**
- Full rank = linearly independent features
- Low rank = redundant/correlated features
- Rank deficiency can cause optimization problems

### 7.2 Trace
Sum of diagonal elements:

```python
trace = np.trace(A)  # sum of A[i,i]
```

**ML applications:**
- Computing matrix norms
- Regularization terms
- Analyzing covariance matrices

### 7.3 Matrix Norms
Measure of matrix "size":

```python
# Frobenius norm (most common)
frobenius_norm = np.linalg.norm(A, 'fro')

# Spectral norm (largest singular value)
spectral_norm = np.linalg.norm(A, 2)

# Nuclear norm (sum of singular values)
nuclear_norm = np.linalg.norm(A, 'nuc')
```

## 8. Eigenvalues and Eigenvectors Preview

For square matrix A, eigenvector v and eigenvalue λ satisfy:
`A @ v = λ @ v`

```python
eigenvalues, eigenvectors = np.linalg.eig(A)
```

**ML applications:**
- Principal Component Analysis (PCA)
- Analyzing neural network dynamics
- Understanding optimization landscapes

*(Detailed coverage in the next post!)*

## 9. Matrix Factorization

Breaking matrices into simpler components:

### 9.1 LU Decomposition
`A = L @ U` (Lower triangular × Upper triangular)

### 9.2 QR Decomposition  
`A = Q @ R` (Orthogonal × Upper triangular)

### 9.3 Singular Value Decomposition (SVD)
`A = U @ S @ V^T`

```python
U, s, Vt = np.linalg.svd(A)
# A = U @ np.diag(s) @ Vt
```

**ML applications:**
- Dimensionality reduction
- Recommender systems
- Image compression
- Pseudoinverse computation

## 10. Computational Considerations

### 10.1 Memory Layout
```python
# Row-major (C-style) vs Column-major (Fortran-style)
A_row_major = np.array([[1, 2, 3], [4, 5, 6]], order='C')
A_col_major = np.array([[1, 2, 3], [4, 5, 6]], order='F')

# Affects performance of operations
```

### 10.2 Broadcasting
NumPy's broadcasting enables operations between different-shaped arrays:

```python
A = np.array([[1, 2, 3],    # Shape: (2, 3)
              [4, 5, 6]])

b = np.array([10, 20, 30])  # Shape: (3,)

# Broadcasting: b is treated as [[10, 20, 30],
#                                [10, 20, 30]]
result = A + b  # Shape: (2, 3)
```

### 10.3 Vectorization Benefits
```python
# Slow: Python loops
result = []
for i in range(len(A)):
    row_result = []
    for j in range(len(B[0])):
        sum_val = 0
        for k in range(len(B)):
            sum_val += A[i][k] * B[k][j]
        row_result.append(sum_val)
    result.append(row_result)

# Fast: Vectorized operations
result = A @ B  # NumPy uses optimized BLAS libraries
```

## 11. Common Matrix Operations in Deep Learning

### 11.1 Batch Normalization
```python
def batch_norm(X, gamma, beta, eps=1e-8):
    # X shape: (batch_size, features)
    mean = np.mean(X, axis=0)  # Shape: (features,)
    var = np.var(X, axis=0)    # Shape: (features,)
    
    # Normalize
    X_norm = (X - mean) / np.sqrt(var + eps)
    
    # Scale and shift
    return gamma * X_norm + beta
```

### 11.2 Attention Mechanism
```python
def attention(Q, K, V):
    # Q, K, V shape: (batch_size, seq_len, d_model)
    d_k = K.shape[-1]
    
    # Compute attention scores
    scores = Q @ K.transpose(-2, -1) / np.sqrt(d_k)
    
    # Apply softmax
    attention_weights = softmax(scores, axis=-1)
    
    # Apply attention to values
    output = attention_weights @ V
    
    return output, attention_weights
```

### 11.3 Convolution as Matrix Multiplication
```python
def convolution_as_matrix(input_matrix, kernel):
    # Convert convolution to matrix multiplication
    # This is how deep learning frameworks optimize convolutions
    
    # Create Toeplitz matrix from kernel
    toeplitz_matrix = create_toeplitz_matrix(kernel)
    
    # Flatten input
    input_vector = input_matrix.flatten()
    
    # Matrix multiplication
    output_vector = toeplitz_matrix @ input_vector
    
    # Reshape to output dimensions
    return output_vector.reshape(output_shape)
```

## 12. Matrix Operations for Different ML Tasks

### 12.1 Regression
```python
# Multiple linear regression
def multiple_regression(X, y):
    # Add bias column
    X_with_bias = np.column_stack([np.ones(len(X)), X])
    
    # Normal equation: θ = (X^T X)^(-1) X^T y
    XtX = X_with_bias.T @ X_with_bias
    Xty = X_with_bias.T @ y
    
    theta = np.linalg.solve(XtX, Xty)
    return theta
```

### 12.2 Classification (Logistic Regression)
```python
def logistic_regression_matrix(X, y, learning_rate=0.01, epochs=1000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    
    for epoch in range(epochs):
        # Forward pass (vectorized)
        z = X @ weights
        predictions = sigmoid(z)
        
        # Compute gradients (vectorized)
        gradient = X.T @ (predictions - y) / n_samples
        
        # Update weights
        weights -= learning_rate * gradient
    
    return weights
```

### 12.3 Dimensionality Reduction (PCA)
```python
def pca_matrix(X, n_components):
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = X_centered.T @ X_centered / (len(X) - 1)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalues (descending)
    idx = np.argsort(eigenvalues)[::-1]
    principal_components = eigenvectors[:, idx[:n_components]]
    
    # Project data
    X_pca = X_centered @ principal_components
    
    return X_pca, principal_components
```

## 13. Performance Tips

### 13.1 Use Appropriate Data Types
```python
# Float32 vs Float64
X_32 = X.astype(np.float32)  # Half the memory, slightly less precision
X_64 = X.astype(np.float64)  # Double precision, more memory
```

### 13.2 Leverage BLAS Libraries
```python
# NumPy automatically uses optimized BLAS when available
# Intel MKL, OpenBLAS, etc.

# Check which BLAS NumPy is using
np.show_config()
```

### 13.3 Memory-Efficient Operations
```python
# In-place operations to save memory
A += B          # Instead of A = A + B
np.add(A, B, out=A)  # Explicit in-place addition

# Use views instead of copies when possible
A_subset = A[start:end]  # View (shares memory)
A_copy = A[start:end].copy()  # Copy (new memory)
```

## 14. Common Mistakes and Debugging

### 14.1 Dimension Mismatches
```python
# Always check shapes before operations
print(f"A shape: {A.shape}, B shape: {B.shape}")

# Use assertions to catch errors early
assert A.shape[1] == B.shape[0], f"Cannot multiply {A.shape} and {B.shape}"
```

### 14.2 Numerical Stability
```python
# Avoid direct matrix inversion for large matrices
# Use solve() instead of inv()

# Bad
theta = np.linalg.inv(X.T @ X) @ X.T @ y

# Good  
theta = np.linalg.solve(X.T @ X, X.T @ y)

# Even better (handles rank deficiency)
theta = np.linalg.lstsq(X, y, rcond=None)[0]
```

### 14.3 Understanding Broadcasting
```python
# Broadcasting can be confusing
A = np.random.randn(100, 50)  # 100 samples, 50 features
b = np.random.randn(50)       # 50-dimensional bias

# This works (broadcasting)
result = A + b  # b is added to each row

# This doesn't work
c = np.random.randn(100)
# result = A + c  # Error: shapes don't align
```

## 15. Conclusion

Matrices are everywhere in machine learning:

1. **Data representation**: Samples × Features
2. **Model parameters**: Weight matrices in neural networks
3. **Transformations**: Linear layers, convolutions, attention
4. **Optimization**: Gradient computations, second-order methods
5. **Analysis**: Eigenanalysis, SVD, matrix factorizations

**Key takeaways:**
- Master basic operations: addition, multiplication, transpose
- Understand broadcasting and vectorization for efficiency
- Use appropriate numerical methods for stability
- Leverage optimized libraries (NumPy, BLAS)
- Always check matrix dimensions before operations

**Next up:** **Eigenvalues and Eigenvectors** - the mathematical tools that power PCA, stability analysis, and understanding the geometric properties of linear transformations!

Matrix operations form the computational backbone of modern machine learning. Every framework, from scikit-learn to PyTorch, relies heavily on efficient matrix computations. Understanding these operations helps you write better code, debug problems faster, and design more efficient algorithms. 