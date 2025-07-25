# Linear Transformations

Linear transformations are the mathematical foundation of countless machine learning operations. Every neural network layer, data preprocessing step, and geometric operation in computer vision can be understood through the lens of linear transformations. This post explores how matrices transform space and why this geometric perspective is crucial for understanding ML algorithms.

## 1. What Are Linear Transformations?

A **linear transformation** is a function T that maps vectors from one space to another while preserving two key properties:

### Mathematical Definition:
For vectors **u**, **v** and scalar **c**:
1. **Additivity**: `T(u + v) = T(u) + T(v)`
2. **Homogeneity**: `T(cu) = cT(u)`

Combined: `T(au + bv) = aT(u) + bT(v)` (**linearity**)

### Matrix Representation:
Every linear transformation can be represented as matrix multiplication:
`T(x) = Ax`

Where **A** is the transformation matrix and **x** is the input vector.

## 2. Fundamental Properties

### 2.1 Preservation of Linear Combinations
```python
import numpy as np

# Define transformation matrix
A = np.array([[2, 1], [1, 3]])

# Test vectors
u = np.array([1, 2])
v = np.array([3, 1])
a, b = 2, -1

# Linearity check
left_side = A @ (a*u + b*v)
right_side = a*(A @ u) + b*(A @ v)

print("Left side:", left_side)
print("Right side:", right_side)
print("Equal?", np.allclose(left_side, right_side))
```

### 2.2 Origin Preservation
Linear transformations always map the origin to itself:
`T(0) = A·0 = 0`

### 2.3 Line Preservation
- Lines through the origin remain lines through the origin
- Parallel lines stay parallel (though they may change direction)
- Ratios along lines are preserved

## 3. Common Linear Transformations in 2D

### 3.1 Identity Transformation
```python
I = np.array([[1, 0], [0, 1]])
# I @ x = x (no change)
```

### 3.2 Scaling
```python
# Uniform scaling by factor s
def scaling_matrix(s):
    return np.array([[s, 0], [0, s]])

# Non-uniform scaling
def scaling_matrix_xy(sx, sy):
    return np.array([[sx, 0], [0, sy]])

# Example: stretch x by 2, shrink y by 0.5
S = scaling_matrix_xy(2, 0.5)
```

### 3.3 Rotation
```python
def rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

# Rotate by 45 degrees
R = rotation_matrix(np.pi/4)
```

### 3.4 Reflection
```python
# Reflect across x-axis
reflect_x = np.array([[1, 0], [0, -1]])

# Reflect across y-axis  
reflect_y = np.array([[-1, 0], [0, 1]])

# Reflect across y=x line
reflect_diag = np.array([[0, 1], [1, 0]])
```

### 3.5 Shearing
```python
# Horizontal shear
def shear_horizontal(k):
    return np.array([[1, k], [0, 1]])

# Vertical shear
def shear_vertical(k):
    return np.array([[1, 0], [k, 1]])
```

### 3.6 Projection
```python
# Project onto x-axis
project_x = np.array([[1, 0], [0, 0]])

# Project onto line y = mx
def project_line(m):
    return (1/(1+m**2)) * np.array([[1, m], [m, m**2]])
```

## 4. Visualizing Transformations

```python
import matplotlib.pyplot as plt

def visualize_transformation(A, title="Transformation"):
    # Create unit square
    square = np.array([[0, 1, 1, 0, 0], 
                       [0, 0, 1, 1, 0]])
    
    # Create unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    
    # Apply transformation
    transformed_square = A @ square
    transformed_circle = A @ circle
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original
    ax1.plot(square[0], square[1], 'b-', linewidth=2, label='Square')
    ax1.plot(circle[0], circle[1], 'r-', linewidth=2, label='Circle')
    ax1.set_title('Original')
    ax1.grid(True)
    ax1.axis('equal')
    ax1.legend()
    
    # Transformed
    ax2.plot(transformed_square[0], transformed_square[1], 'b-', linewidth=2, label='Transformed Square')
    ax2.plot(transformed_circle[0], transformed_circle[1], 'r-', linewidth=2, label='Transformed Circle')
    ax2.set_title(f'After {title}')
    ax2.grid(True)
    ax2.axis('equal')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# Example usage
A_scale = scaling_matrix_xy(2, 0.5)
visualize_transformation(A_scale, "Scaling")

A_rotation = rotation_matrix(np.pi/6)
visualize_transformation(A_rotation, "Rotation")
```

## 5. Composition of Transformations

Multiple transformations can be combined by matrix multiplication:

```python
# First scale, then rotate, then translate
def composite_transformation():
    # Scale by 1.5
    S = scaling_matrix(1.5)
    
    # Rotate by 30 degrees
    R = rotation_matrix(np.pi/6)
    
    # Composite: apply S first, then R
    # Note: matrix multiplication is right-to-left
    composite = R @ S
    
    return composite, S, R

composite, S, R = composite_transformation()

# Test on a point
point = np.array([1, 1])
result1 = composite @ point  # Direct composite
result2 = R @ (S @ point)    # Step by step

print("Composite result:", result1)
print("Step-by-step result:", result2)
print("Equal?", np.allclose(result1, result2))
```

## 6. Linear Transformations in Machine Learning

### 6.1 Neural Network Layers
Every fully connected layer is a linear transformation followed by a non-linear activation:

```python
def linear_layer(X, W, b):
    """
    Linear transformation in neural networks
    
    X: input data (batch_size, input_features)
    W: weight matrix (input_features, output_features)
    b: bias vector (output_features,)
    """
    # Linear transformation
    linear_output = X @ W + b
    
    # Note: bias addition makes this an "affine" transformation,
    # not strictly linear, but the core operation is linear
    
    return linear_output

# Example: MNIST input to hidden layer
batch_size = 32
input_features = 784  # 28x28 images
hidden_features = 128

X = np.random.randn(batch_size, input_features)
W = np.random.randn(input_features, hidden_features) * 0.1
b = np.zeros(hidden_features)

hidden = linear_layer(X, W, b)
print(f"Input shape: {X.shape}")
print(f"Output shape: {hidden.shape}")
```

### 6.2 Principal Component Analysis (PCA)
PCA finds the best linear transformation to reduce dimensionality:

```python
def pca_transformation(X, n_components):
    """
    PCA as a linear transformation
    """
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Find eigenvectors (principal components)
    eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalue magnitude
    idx = np.argsort(eigenvals)[::-1]
    eigenvecs = eigenvecs[:, idx]
    
    # Create transformation matrix
    W_pca = eigenvecs[:, :n_components]
    
    # Apply transformation
    X_transformed = X_centered @ W_pca
    
    return X_transformed, W_pca

# Example
np.random.seed(42)
X = np.random.randn(100, 5)
X_pca, W_pca = pca_transformation(X, 2)

print(f"Original shape: {X.shape}")
print(f"Transformed shape: {X_pca.shape}")
print(f"Transformation matrix shape: {W_pca.shape}")
```

### 6.3 Feature Scaling and Normalization
Common preprocessing steps are linear transformations:

```python
def standardization_matrix(X):
    """Create standardization transformation matrix"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    
    # Transformation: (x - mean) / std
    # This is affine: T(x) = Dx - Dm where D = diag(1/std)
    
    D = np.diag(1 / std)
    offset = D @ mean
    
    return D, offset

def min_max_scaling_matrix(X):
    """Create min-max scaling transformation matrix"""
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    
    # Transformation: (x - min) / (max - min)
    scale = 1 / (max_vals - min_vals)
    D = np.diag(scale)
    offset = D @ min_vals
    
    return D, offset
```

### 6.4 Data Augmentation in Computer Vision
Geometric transformations for data augmentation:

```python
def create_augmentation_matrices():
    """Common image augmentation transformations"""
    
    transformations = {
        'horizontal_flip': np.array([[-1, 0], [0, 1]]),
        'vertical_flip': np.array([[1, 0], [0, -1]]),
        'rotate_90': np.array([[0, -1], [1, 0]]),
        'rotate_180': np.array([[-1, 0], [0, -1]]),
        'rotate_270': np.array([[0, 1], [-1, 0]]),
    }
    
    return transformations

def apply_random_rotation(image_coords, max_angle=30):
    """Apply random rotation for data augmentation"""
    angle = np.random.uniform(-max_angle, max_angle) * np.pi / 180
    R = rotation_matrix(angle)
    
    return R @ image_coords
```

## 7. Advanced Concepts

### 7.1 Change of Basis
Linear transformations can represent changes between coordinate systems:

```python
def change_of_basis_example():
    """Change from standard basis to new basis"""
    
    # New basis vectors (columns of P)
    v1 = np.array([1, 1])
    v2 = np.array([1, -1])
    P = np.column_stack([v1, v2])
    
    # Vector in standard coordinates
    x_standard = np.array([3, 1])
    
    # Convert to new basis
    x_new_basis = np.linalg.solve(P, x_standard)
    
    # Convert back to standard basis
    x_back = P @ x_new_basis
    
    print(f"Original: {x_standard}")
    print(f"In new basis: {x_new_basis}")
    print(f"Back to standard: {x_back}")
    print(f"Recovered correctly: {np.allclose(x_standard, x_back)}")

change_of_basis_example()
```

### 7.2 Kernel and Image of Transformations

```python
def analyze_transformation(A):
    """Analyze kernel (null space) and image (column space) of transformation"""
    
    # Kernel (null space): vectors that map to zero
    U, s, Vt = np.linalg.svd(A)
    kernel_basis = Vt[s < 1e-10]  # Vectors with tiny singular values
    
    # Image (column space): span of columns
    image_basis = U[:, s >= 1e-10]
    
    # Rank and nullity
    rank = np.sum(s >= 1e-10)
    nullity = A.shape[1] - rank
    
    print(f"Matrix shape: {A.shape}")
    print(f"Rank: {rank}")
    print(f"Nullity: {nullity}")
    print(f"Kernel dimension: {kernel_basis.shape[0]}")
    print(f"Image dimension: {image_basis.shape[1]}")
    
    return kernel_basis, image_basis

# Example: singular matrix
A_singular = np.array([[1, 2], [2, 4]])  # Rank 1
kernel, image = analyze_transformation(A_singular)
```

## 8. Non-Linear Transformations in ML

While this post focuses on linear transformations, it's worth noting where non-linearity comes in:

### 8.1 Activation Functions
```python
def neural_network_layer(x, W, b, activation='relu'):
    """Neural network layer: linear transformation + non-linear activation"""
    
    # Linear transformation
    z = x @ W + b
    
    # Non-linear activation
    if activation == 'relu':
        return np.maximum(0, z)
    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(-z))
    elif activation == 'tanh':
        return np.tanh(z)
    else:
        return z  # Linear activation
```

### 8.2 Kernel Methods
```python
def polynomial_kernel_transformation(X, degree=2):
    """Transform data to higher-dimensional space (polynomial kernel)"""
    
    # For 2D input, degree-2 polynomial kernel creates features:
    # [1, x1, x2, x1^2, x1*x2, x2^2]
    
    n_samples, n_features = X.shape
    if n_features == 2:
        x1, x2 = X[:, 0], X[:, 1]
        
        # Create polynomial features
        phi_X = np.column_stack([
            np.ones(n_samples),  # bias
            x1, x2,              # linear terms
            x1**2, x1*x2, x2**2  # quadratic terms
        ])
        
        return phi_X
    
    # For general case, use sklearn
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=degree)
    return poly.fit_transform(X)
```

## 9. Computational Considerations

### 9.1 Efficient Matrix Operations
```python
def efficient_transformations():
    """Tips for efficient linear transformations"""
    
    # Use appropriate data types
    X_float32 = np.random.randn(1000, 100).astype(np.float32)
    W_float32 = np.random.randn(100, 50).astype(np.float32)
    
    # Leverage BLAS libraries through NumPy
    result = X_float32 @ W_float32  # Optimized
    
    # Batch operations when possible
    batch_size = 32
    X_batch = np.random.randn(batch_size, 100)
    
    # Process entire batch at once (vectorized)
    batch_result = X_batch @ W_float32  # Much faster than loops
    
    return result, batch_result
```

### 9.2 Memory-Efficient Transformations
```python
def memory_efficient_transformation(X, W, chunk_size=1000):
    """Apply transformation in chunks to save memory"""
    
    n_samples = X.shape[0]
    n_output = W.shape[1]
    result = np.zeros((n_samples, n_output))
    
    for i in range(0, n_samples, chunk_size):
        end_idx = min(i + chunk_size, n_samples)
        result[i:end_idx] = X[i:end_idx] @ W
    
    return result
```

## 10. Geometric Intuition for ML Algorithms

### 10.1 Linear Regression as Projection
```python
def linear_regression_projection(X, y):
    """Understanding linear regression as projection onto column space"""
    
    # Normal equation: θ = (X^T X)^(-1) X^T y
    theta = np.linalg.solve(X.T @ X, X.T @ y)
    
    # Prediction: project y onto column space of X
    y_pred = X @ theta
    
    # Residual: component orthogonal to column space
    residual = y - y_pred
    
    # Verify orthogonality: residual ⊥ column space of X
    orthogonality_check = X.T @ residual
    
    print(f"Orthogonality check (should be ~0): {np.max(np.abs(orthogonality_check))}")
    
    return theta, y_pred, residual
```

### 10.2 Support Vector Machines and Transformations
```python
def svm_transformation_intuition():
    """How SVM uses linear transformations"""
    
    # SVM finds hyperplane: w^T x + b = 0
    # This is equivalent to: w^T (x - x0) = 0 for point x0 on hyperplane
    
    # The transformation w^T projects points onto the normal direction
    # Decision boundary is where this projection equals -b/||w||
    
    # Example: 2D case
    w = np.array([1, 1])  # Normal vector
    b = -1
    
    # Generate test points
    X_test = np.array([[0, 2], [1, 1], [2, 0]])
    
    # Project onto normal direction
    projections = X_test @ w
    
    # Classify: sign(w^T x + b)
    classifications = np.sign(projections + b)
    
    print("Points:", X_test)
    print("Projections:", projections)
    print("Classifications:", classifications)
```

## 11. Debugging Linear Transformations

### 11.1 Common Issues and Solutions
```python
def debug_transformation(A, x):
    """Debug linear transformation issues"""
    
    print(f"Transformation matrix shape: {A.shape}")
    print(f"Input vector shape: {x.shape}")
    
    # Check dimension compatibility
    if A.shape[1] != x.shape[0]:
        print("ERROR: Dimension mismatch!")
        return None
    
    # Check for numerical issues
    cond_num = np.linalg.cond(A)
    if cond_num > 1e12:
        print(f"WARNING: Matrix is ill-conditioned (cond={cond_num:.2e})")
    
    # Check for special properties
    if np.allclose(A, A.T):
        print("Matrix is symmetric")
    
    det_A = np.linalg.det(A) if A.shape[0] == A.shape[1] else "N/A"
    print(f"Determinant: {det_A}")
    
    # Apply transformation
    try:
        result = A @ x
        print(f"Transformation successful, output shape: {result.shape}")
        return result
    except Exception as e:
        print(f"ERROR: {e}")
        return None
```

### 11.2 Visualizing High-Dimensional Transformations
```python
def visualize_high_dim_transformation(A, n_samples=1000):
    """Visualize effect of high-dimensional transformation"""
    
    input_dim = A.shape[1]
    output_dim = A.shape[0]
    
    # Generate random input points
    X = np.random.randn(n_samples, input_dim)
    
    # Apply transformation
    Y = X @ A.T  # Shape: (n_samples, output_dim)
    
    # Analyze transformation effects
    input_stats = {
        'mean': np.mean(X, axis=0),
        'std': np.std(X, axis=0),
        'norm': np.mean(np.linalg.norm(X, axis=1))
    }
    
    output_stats = {
        'mean': np.mean(Y, axis=0),
        'std': np.std(Y, axis=0),
        'norm': np.mean(np.linalg.norm(Y, axis=1))
    }
    
    print(f"Input dimension: {input_dim}, Output dimension: {output_dim}")
    print(f"Average input norm: {input_stats['norm']:.3f}")
    print(f"Average output norm: {output_stats['norm']:.3f}")
    print(f"Transformation scaling factor: {output_stats['norm']/input_stats['norm']:.3f}")
    
    return X, Y, input_stats, output_stats
```

## 12. Applications in Modern ML

### 12.1 Transformer Attention as Linear Transformation
```python
def attention_linear_transformations(X, d_model=512, n_heads=8):
    """Attention mechanism uses multiple linear transformations"""
    
    batch_size, seq_len, _ = X.shape
    d_k = d_model // n_heads
    
    # Query, Key, Value projections (linear transformations)
    W_q = np.random.randn(d_model, d_model) / np.sqrt(d_model)
    W_k = np.random.randn(d_model, d_model) / np.sqrt(d_model)
    W_v = np.random.randn(d_model, d_model) / np.sqrt(d_model)
    
    # Apply linear transformations
    Q = X @ W_q  # Shape: (batch_size, seq_len, d_model)
    K = X @ W_k
    V = X @ W_v
    
    # Reshape for multi-head attention
    Q = Q.reshape(batch_size, seq_len, n_heads, d_k)
    K = K.reshape(batch_size, seq_len, n_heads, d_k)
    V = V.reshape(batch_size, seq_len, n_heads, d_k)
    
    print(f"Input shape: {X.shape}")
    print(f"Q, K, V shapes: {Q.shape}")
    
    return Q, K, V
```

### 12.2 Convolutional Layers as Structured Linear Transformations
```python
def convolution_as_linear_transformation():
    """Understanding convolution as a structured linear transformation"""
    
    # For a 1D convolution, we can represent it as matrix multiplication
    # with a Toeplitz matrix
    
    def create_toeplitz_matrix(kernel, input_length):
        """Create convolution matrix (Toeplitz structure)"""
        kernel_size = len(kernel)
        output_length = input_length - kernel_size + 1
        
        conv_matrix = np.zeros((output_length, input_length))
        
        for i in range(output_length):
            conv_matrix[i, i:i+kernel_size] = kernel
        
        return conv_matrix
    
    # Example
    kernel = np.array([1, -1, 1])  # Edge detection kernel
    input_length = 7
    
    conv_matrix = create_toeplitz_matrix(kernel, input_length)
    
    print("Convolution matrix:")
    print(conv_matrix)
    
    # Test with sample input
    x = np.array([1, 2, 3, 4, 3, 2, 1])
    
    # Convolution as matrix multiplication
    y_matrix = conv_matrix @ x
    
    # Direct convolution
    y_direct = np.convolve(x, kernel, mode='valid')
    
    print(f"Matrix result: {y_matrix}")
    print(f"Direct convolution: {y_direct}")
    print(f"Equal: {np.allclose(y_matrix, y_direct)}")
```

## 13. Conclusion

Linear transformations are the building blocks of machine learning:

### Key Insights:
1. **Geometric Understanding**: Transformations stretch, rotate, reflect, and project space
2. **Composition Power**: Complex transformations are built from simple ones
3. **Preservation Properties**: Lines, parallelism, and ratios are preserved
4. **Computational Efficiency**: Matrix operations enable vectorized computations

### ML Applications:
- **Neural Networks**: Every linear layer is a transformation
- **Feature Engineering**: Scaling, normalization, PCA
- **Computer Vision**: Rotations, translations, augmentations
- **Dimensionality Reduction**: Projections onto lower-dimensional spaces

### Practical Guidelines:
- Visualize 2D/3D transformations to build intuition
- Check matrix properties (rank, condition number) for stability
- Use appropriate numerical methods for large-scale problems
- Compose transformations through matrix multiplication

**Understanding linear transformations gives you:**
- Geometric intuition for ML algorithms
- Better debugging capabilities
- Insights into why certain methods work
- Foundation for advanced topics like manifold learning

Linear transformations bridge the gap between abstract mathematics and practical machine learning, providing both computational tools and geometric insights that are essential for modern AI development.

Remember: while many ML operations involve non-linear functions, the linear components often dominate computational cost and provide the primary structure for learning algorithms. Master linear transformations, and you'll have a powerful lens for understanding machine learning! 