# Eigenvalues and Eigenvectors

Eigenvalues and eigenvectors are among the most powerful concepts in linear algebra, with profound applications in machine learning. They're the mathematical foundation behind Principal Component Analysis (PCA), Google's PageRank algorithm, stability analysis of neural networks, and much more. Understanding them unlocks deeper insights into how data behaves and why certain ML techniques work.

## 1. What Are Eigenvalues and Eigenvectors?

For a square matrix A, an **eigenvector** v is a non-zero vector that doesn't change direction when A is applied to it. It only gets scaled by a factor called the **eigenvalue** λ.

### Mathematical Definition:
`A v = λ v`

Where:
- `A` = square matrix
- `v` = eigenvector (non-zero vector)
- `λ` = eigenvalue (scalar)

### Intuitive Understanding:
Think of a transformation matrix A as a function that stretches, shrinks, or rotates vectors. Most vectors change both magnitude and direction. But eigenvectors are special - they only change in magnitude (by the eigenvalue), not direction.

## 2. Simple Example

Let's find eigenvalues and eigenvectors for:
```
A = [3  1]
    [0  2]
```

**Step 1:** Solve the characteristic equation `det(A - λI) = 0`

```
A - λI = [3-λ   1 ] 
         [ 0   2-λ]

det(A - λI) = (3-λ)(2-λ) - 1×0 = (3-λ)(2-λ) = 0
```

**Step 2:** Find eigenvalues
`λ₁ = 3` and `λ₂ = 2`

**Step 3:** Find eigenvectors by solving `(A - λI)v = 0`

For λ₁ = 3:
```
(A - 3I)v = [0  1] [v₁] = [0]
            [0 -1] [v₂]   [0]

This gives v₂ = 0, so v₁ = [1, 0]ᵀ
```

For λ₂ = 2:
```
(A - 2I)v = [1  1] [v₁] = [0]
            [0  0] [v₂]   [0]

This gives v₁ + v₂ = 0, so v₂ = [1, -1]ᵀ
```

**Verification:**
```python
import numpy as np

A = np.array([[3, 1], [0, 2]])
v1 = np.array([1, 0])
v2 = np.array([1, -1])

print(A @ v1)  # [3, 0] = 3 * [1, 0] ✓
print(A @ v2)  # [2, -2] = 2 * [1, -1] ✓
```

## 3. Computing Eigenvalues and Eigenvectors

### 3.1 Using NumPy
```python
import numpy as np

A = np.array([[3, 1], [0, 2]])
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:")
print(eigenvectors)
```

### 3.2 Manual Computation Steps
```python
def find_eigenvalues_2x2(A):
    """Find eigenvalues for 2x2 matrix using characteristic polynomial"""
    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]
    
    # Characteristic polynomial: λ² - (a+d)λ + (ad-bc) = 0
    trace = a + d
    determinant = a*d - b*c
    
    # Quadratic formula
    discriminant = trace**2 - 4*determinant
    lambda1 = (trace + np.sqrt(discriminant)) / 2
    lambda2 = (trace - np.sqrt(discriminant)) / 2
    
    return lambda1, lambda2

def find_eigenvector(A, eigenvalue):
    """Find eigenvector for given eigenvalue"""
    n = A.shape[0]
    B = A - eigenvalue * np.eye(n)
    
    # Find null space of B
    # For 2x2, we can solve manually
    if n == 2:
        if abs(B[0, 0]) > 1e-10:
            return np.array([B[0, 1], -B[0, 0]])
        else:
            return np.array([1, 0])
    
    # For larger matrices, use SVD
    _, _, V = np.linalg.svd(B)
    return V[-1]  # Last row of V
```

## 4. Geometric Interpretation

Eigenvalues and eigenvectors reveal the fundamental behavior of linear transformations:

### 4.1 Stretching/Shrinking
- `λ > 1`: Eigenvector gets stretched
- `0 < λ < 1`: Eigenvector gets shrunk  
- `λ < 0`: Eigenvector flips direction
- `λ = 0`: Eigenvector maps to zero (matrix is singular)

### 4.2 Visualization Example
```python
import matplotlib.pyplot as plt

# Create a transformation matrix
A = np.array([[2, 1], [1, 2]])

# Generate some vectors
theta = np.linspace(0, 2*np.pi, 100)
unit_circle = np.array([np.cos(theta), np.sin(theta)])

# Apply transformation
transformed = A @ unit_circle

# Plot original and transformed
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(unit_circle[0], unit_circle[1], 'b-', label='Original')
plt.axis('equal')
plt.title('Unit Circle')

plt.subplot(1, 2, 2)
plt.plot(transformed[0], transformed[1], 'r-', label='Transformed')
plt.axis('equal')
plt.title('After Transformation')

# Add eigenvectors
eigenvals, eigenvecs = np.linalg.eig(A)
for i, (val, vec) in enumerate(zip(eigenvals, eigenvecs.T)):
    plt.arrow(0, 0, vec[0]*val, vec[1]*val, 
              head_width=0.1, color=f'C{i}', 
              label=f'λ={val:.1f}')

plt.legend()
plt.show()
```

## 5. Principal Component Analysis (PCA)

PCA is probably the most famous application of eigenvalues/eigenvectors in ML.

### 5.1 The Idea
Find the directions (principal components) along which data varies the most.

### 5.2 Algorithm
```python
def pca(X, n_components=None):
    """
    Perform PCA using eigendecomposition
    
    X: data matrix (samples × features)
    n_components: number of components to keep
    """
    # Step 1: Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Step 2: Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Step 3: Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Step 4: Sort by eigenvalues (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    if n_components:
        eigenvalues = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]
    
    # Step 5: Transform data
    X_pca = X_centered @ eigenvectors
    
    return X_pca, eigenvalues, eigenvectors

# Example usage
np.random.seed(42)
X = np.random.randn(100, 3)
X[:, 1] = X[:, 0] + 0.1 * np.random.randn(100)  # Correlated features

X_pca, eigenvals, eigenvecs = pca(X, n_components=2)

print("Explained variance ratio:", eigenvals / np.sum(eigenvals))
```

### 5.3 Why Eigenvalues Work for PCA
- **Covariance matrix** shows how features vary together
- **Eigenvectors** of covariance matrix are the principal component directions
- **Eigenvalues** indicate how much variance is explained by each component
- Largest eigenvalue → most important direction of variation

## 6. Applications in Machine Learning

### 6.1 Dimensionality Reduction
```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

# Load high-dimensional data
digits = load_digits()
X = digits.data  # 64 features (8x8 pixel images)

# Apply PCA
pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X)

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_reduced.shape}")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
```

### 6.2 Face Recognition (Eigenfaces)
```python
def eigenfaces_demo():
    """Simplified eigenfaces for face recognition"""
    # Each face is a vector of pixel intensities
    # Faces matrix: (n_faces, n_pixels)
    
    # Step 1: Compute mean face
    mean_face = np.mean(faces, axis=0)
    
    # Step 2: Center faces
    centered_faces = faces - mean_face
    
    # Step 3: Compute covariance matrix
    # For computational efficiency, use faces @ faces.T instead of faces.T @ faces
    cov = centered_faces @ centered_faces.T
    
    # Step 4: Eigendecomposition
    eigenvals, eigenvecs = np.linalg.eig(cov)
    
    # Step 5: Get actual eigenfaces
    eigenfaces = centered_faces.T @ eigenvecs
    
    # Step 6: Normalize eigenfaces
    for i in range(eigenfaces.shape[1]):
        eigenfaces[:, i] /= np.linalg.norm(eigenfaces[:, i])
    
    return eigenfaces, mean_face
```

### 6.3 Graph Analysis (PageRank)
```python
def pagerank_eigenvector(adjacency_matrix, damping=0.85, max_iter=100):
    """
    Compute PageRank using the dominant eigenvector
    
    PageRank finds the eigenvector of the transition matrix
    corresponding to eigenvalue 1
    """
    n = adjacency_matrix.shape[0]
    
    # Create transition matrix
    # Add damping for numerical stability
    transition_matrix = damping * adjacency_matrix / adjacency_matrix.sum(axis=1, keepdims=True)
    transition_matrix += (1 - damping) / n * np.ones((n, n))
    
    # Power iteration to find dominant eigenvector
    pagerank = np.ones(n) / n
    
    for _ in range(max_iter):
        pagerank_new = transition_matrix.T @ pagerank
        if np.linalg.norm(pagerank_new - pagerank) < 1e-8:
            break
        pagerank = pagerank_new
    
    return pagerank / pagerank.sum()
```

### 6.4 Stability Analysis of Neural Networks
```python
def analyze_network_stability(weight_matrix):
    """
    Analyze if a recurrent network is stable
    
    Network is stable if all eigenvalues have magnitude < 1
    """
    eigenvalues = np.linalg.eigvals(weight_matrix)
    
    # Check stability
    max_eigenvalue_magnitude = np.max(np.abs(eigenvalues))
    is_stable = max_eigenvalue_magnitude < 1
    
    print(f"Maximum eigenvalue magnitude: {max_eigenvalue_magnitude:.3f}")
    print(f"Network is {'stable' if is_stable else 'unstable'}")
    
    return eigenvalues, is_stable

# Example: Simple RNN weight matrix
W_rnn = np.random.randn(10, 10) * 0.1  # Small weights for stability
eigenvals, stable = analyze_network_stability(W_rnn)
```

## 7. Eigendecomposition and Matrix Powers

One powerful property: if we know eigenvalues and eigenvectors, we can easily compute matrix powers.

### 7.1 Diagonalization
If A has linearly independent eigenvectors:
`A = P D P⁻¹`

Where:
- P = matrix of eigenvectors
- D = diagonal matrix of eigenvalues

### 7.2 Computing Matrix Powers
```python
def matrix_power_eigen(A, n):
    """Compute A^n using eigendecomposition"""
    eigenvals, eigenvecs = np.linalg.eig(A)
    
    # A = P D P^(-1)
    # A^n = P D^n P^(-1)
    D_n = np.diag(eigenvals ** n)
    A_n = eigenvecs @ D_n @ np.linalg.inv(eigenvecs)
    
    return A_n

# Example: Fibonacci sequence using matrix powers
def fibonacci_eigen(n):
    """Compute nth Fibonacci number using eigendecomposition"""
    # Fibonacci recurrence matrix
    F = np.array([[1, 1], [1, 0]])
    
    if n == 0:
        return 0
    
    F_n = matrix_power_eigen(F, n)
    return int(F_n[0, 1])  # Extract Fibonacci number

print([fibonacci_eigen(i) for i in range(10)])
```

## 8. Symmetric vs Non-Symmetric Matrices

### 8.1 Symmetric Matrices
Special properties when A = Aᵀ:
- All eigenvalues are real
- Eigenvectors are orthogonal
- Always diagonalizable

```python
def is_symmetric(A, tol=1e-10):
    return np.allclose(A, A.T, atol=tol)

# Covariance matrices are always symmetric
X = np.random.randn(100, 5)
cov_matrix = np.cov(X.T)
print(f"Is covariance matrix symmetric? {is_symmetric(cov_matrix)}")

eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
print(f"Are eigenvalues real? {np.all(np.isreal(eigenvals))}")

# Check orthogonality of eigenvectors
dot_product = eigenvecs[:, 0] @ eigenvecs[:, 1]
print(f"Dot product of first two eigenvectors: {dot_product:.10f}")
```

### 8.2 Non-Symmetric Matrices
Can have complex eigenvalues and eigenvectors:

```python
# Example: rotation matrix
theta = np.pi / 4
rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta),  np.cos(theta)]])

eigenvals, eigenvecs = np.linalg.eig(rotation_matrix)
print("Eigenvalues:", eigenvals)
print("Are eigenvalues complex?", np.any(np.iscomplex(eigenvals)))
```

## 9. Practical Considerations

### 9.1 Numerical Stability
```python
def safe_eigendecomposition(A, check_condition=True):
    """Eigendecomposition with numerical checks"""
    
    if check_condition:
        condition_number = np.linalg.cond(A)
        if condition_number > 1e12:
            print(f"Warning: Matrix is ill-conditioned (cond={condition_number:.2e})")
    
    try:
        eigenvals, eigenvecs = np.linalg.eig(A)
        
        # Sort by eigenvalue magnitude (descending)
        idx = np.argsort(np.abs(eigenvals))[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        return eigenvals, eigenvecs
        
    except np.linalg.LinAlgError:
        print("Eigendecomposition failed - using SVD instead")
        U, s, Vt = np.linalg.svd(A)
        return s, U
```

### 9.2 Large Matrix Considerations
For very large matrices, computing all eigenvalues is expensive:

```python
from scipy.sparse.linalg import eigs

def top_k_eigenvalues(A, k=5):
    """Find top k eigenvalues efficiently"""
    # For large sparse matrices
    eigenvals, eigenvecs = eigs(A, k=k, which='LM')  # Largest Magnitude
    
    # Sort by eigenvalue magnitude
    idx = np.argsort(np.abs(eigenvals))[::-1]
    return eigenvals[idx], eigenvecs[:, idx]
```

### 9.3 Choosing Number of Components in PCA
```python
def choose_pca_components(eigenvalues, variance_threshold=0.95):
    """Choose number of components to retain given % of variance"""
    cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    print(f"Components needed for {variance_threshold*100}% variance: {n_components}")
    return n_components

# Example
eigenvals = np.array([50, 30, 15, 3, 1, 0.5, 0.3, 0.2])
n_comp = choose_pca_components(eigenvals, 0.9)
```

## 10. Advanced Applications

### 10.1 Spectral Clustering
```python
def spectral_clustering_demo(adjacency_matrix, n_clusters=2):
    """Spectral clustering using eigendecomposition"""
    
    # Compute degree matrix
    degree_matrix = np.diag(adjacency_matrix.sum(axis=1))
    
    # Compute Laplacian matrix
    laplacian = degree_matrix - adjacency_matrix
    
    # Normalized Laplacian (better numerical properties)
    sqrt_deg_inv = np.diag(1.0 / np.sqrt(adjacency_matrix.sum(axis=1)))
    normalized_laplacian = sqrt_deg_inv @ laplacian @ sqrt_deg_inv
    
    # Find smallest eigenvalues/eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(normalized_laplacian)
    
    # Use smallest k eigenvectors for clustering
    clustering_features = eigenvecs[:, :n_clusters]
    
    # Apply k-means to the eigenspace
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(clustering_features)
    
    return cluster_labels
```

### 10.2 Markov Chain Analysis
```python
def analyze_markov_chain(transition_matrix):
    """Analyze steady-state behavior of Markov chain"""
    
    eigenvals, eigenvecs = np.linalg.eig(transition_matrix.T)
    
    # Find eigenvalue closest to 1 (steady state)
    steady_state_idx = np.argmin(np.abs(eigenvals - 1))
    steady_state_eigval = eigenvals[steady_state_idx]
    steady_state_vector = eigenvecs[:, steady_state_idx].real
    
    # Normalize to probability distribution
    steady_state_vector = steady_state_vector / steady_state_vector.sum()
    
    print(f"Steady state eigenvalue: {steady_state_eigval}")
    print(f"Steady state distribution: {steady_state_vector}")
    
    return steady_state_vector
```

## 11. Relationship to Singular Value Decomposition (SVD)

For non-square matrices, SVD generalizes eigendecomposition:

```python
def compare_eigen_svd(A):
    """Compare eigendecomposition and SVD for square matrices"""
    
    # Eigendecomposition
    eigenvals_A, eigenvecs_A = np.linalg.eig(A)
    
    # SVD
    U, s, Vt = np.linalg.svd(A)
    
    print("Eigenvalues:", np.sort(eigenvals_A)[::-1])
    print("Singular values:", s)
    
    # For symmetric positive definite matrices:
    # eigenvalues = singular values
    # eigenvectors = U = V
    
    if is_symmetric(A) and np.all(eigenvals_A > 0):
        print("Matrix is symmetric positive definite")
        print("Eigenvalues ≈ Singular values:", 
              np.allclose(np.sort(np.abs(eigenvals_A))[::-1], s))
```

## 12. Common Pitfalls and Debugging

### 12.1 Complex Eigenvalues
```python
def handle_complex_eigenvalues(eigenvals, eigenvecs):
    """Handle complex eigenvalues in real applications"""
    
    if np.any(np.iscomplex(eigenvals)):
        print("Warning: Complex eigenvalues detected")
        
        # For real matrices, complex eigenvalues come in conjugate pairs
        # Often we only need the real parts or magnitudes
        
        real_parts = eigenvals.real
        imaginary_parts = eigenvals.imag
        magnitudes = np.abs(eigenvals)
        
        return {
            'real_parts': real_parts,
            'imaginary_parts': imaginary_parts, 
            'magnitudes': magnitudes,
            'eigenvectors': eigenvecs
        }
    
    return {'eigenvalues': eigenvals, 'eigenvectors': eigenvecs}
```

### 12.2 Zero Eigenvalues
```python
def analyze_zero_eigenvalues(A, tol=1e-10):
    """Analyze matrices with zero eigenvalues"""
    
    eigenvals, eigenvecs = np.linalg.eig(A)
    
    # Find near-zero eigenvalues
    zero_mask = np.abs(eigenvals) < tol
    zero_eigenvals = eigenvals[zero_mask]
    zero_eigenvecs = eigenvecs[:, zero_mask]
    
    if np.any(zero_mask):
        print(f"Found {np.sum(zero_mask)} near-zero eigenvalues")
        print("Matrix is singular or nearly singular")
        print("Null space dimension:", np.sum(zero_mask))
        
        # These eigenvectors span the null space
        return zero_eigenvecs
    
    return None
```

## 13. Conclusion

Eigenvalues and eigenvectors are fundamental tools that reveal the intrinsic properties of linear transformations:

### Key Applications in ML:
1. **PCA**: Dimensionality reduction using dominant eigenvectors
2. **Spectral Clustering**: Clustering using graph Laplacian eigenvectors  
3. **PageRank**: Web page ranking using dominant eigenvector
4. **Stability Analysis**: Network stability via eigenvalue magnitudes
5. **Markov Chains**: Steady states as eigenproblems

### Important Properties:
- **Geometric meaning**: Directions that don't change under transformation
- **Computational power**: Enable efficient matrix powers and exponentials
- **Data insights**: Reveal principal directions of variation
- **System analysis**: Characterize long-term behavior

### Practical Guidelines:
- Use appropriate numerical methods for large/sparse matrices
- Check for numerical stability issues
- Understand symmetric vs non-symmetric matrix properties
- Choose components based on eigenvalue magnitudes

**Next up:** **Linear Transformations** - how matrices transform space and why this matters for understanding machine learning algorithms geometrically!

Understanding eigenvalues and eigenvectors gives you powerful tools for analyzing data, designing algorithms, and understanding why certain machine learning techniques work. They bridge the gap between abstract mathematics and practical ML applications. 