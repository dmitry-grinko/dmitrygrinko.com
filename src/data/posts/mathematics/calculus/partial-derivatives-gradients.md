# Partial Derivatives and Gradients

While regular derivatives work with functions of a single variable, most machine learning problems involve functions with multiple variables. This is where partial derivatives and gradients become essential tools for optimization and understanding model behavior.

## 1. From Single to Multiple Variables

### Single Variable Function:
`f(x) = x²` → derivative: `f'(x) = 2x`

### Multiple Variable Function:
`f(x, y) = x² + y²` → Now what?

This is where **partial derivatives** come to the rescue!

## 2. What Are Partial Derivatives?

A partial derivative measures how a function changes with respect to **one variable while keeping all other variables constant**.

### Notation:
- `∂f/∂x` → partial derivative with respect to x
- `∂f/∂y` → partial derivative with respect to y

### Example:
For `f(x, y) = x² + 3xy + y²`:
- `∂f/∂x = 2x + 3y` (treat y as a constant)
- `∂f/∂y = 3x + 2y` (treat x as a constant)

## 3. Computing Partial Derivatives

### 3.1 Basic Rules
Same rules as regular derivatives, but treat other variables as constants:

**Example 1:** `f(x, y) = x³y²`
- `∂f/∂x = 3x²y²` (y² is treated as constant)
- `∂f/∂y = 2x³y` (x³ is treated as constant)

**Example 2:** `f(x, y, z) = x²z + y³ - 5z`
- `∂f/∂x = 2xz`
- `∂f/∂y = 3y²`
- `∂f/∂z = x² - 5`

### 3.2 Chain Rule for Multiple Variables
For composite functions: `f(g(x, y), h(x, y))`

`∂f/∂x = (∂f/∂g)(∂g/∂x) + (∂f/∂h)(∂h/∂x)`

## 4. The Gradient Vector

The **gradient** is a vector containing all partial derivatives of a function.

### Definition:
For function `f(x, y, z)`, the gradient is:

`∇f = [∂f/∂x, ∂f/∂y, ∂f/∂z]`

### Example:
`f(x, y) = x² + y²`
`∇f = [2x, 2y]`

At point (3, 4): `∇f(3,4) = [6, 8]`

## 5. Geometric Interpretation

### 5.1 Direction of Steepest Ascent
The gradient vector points in the direction of **steepest increase** of the function.

### 5.2 Perpendicular to Contour Lines
Gradient vectors are always perpendicular to the contour lines (level curves) of the function.

### 5.3 Magnitude = Rate of Change
The length of the gradient vector tells us how steep the function is at that point.

## 6. Gradients in Machine Learning

### 6.1 Loss Function Optimization
In ML, we typically want to **minimize** a loss function `L(w₁, w₂, ..., wₙ)` where `w` represents model weights.

The gradient tells us:
- **Direction**: Which way to adjust weights
- **Magnitude**: How much to adjust weights

### 6.2 Gradient Descent Algorithm
```python
# Basic gradient descent
def gradient_descent(loss_function, initial_weights, learning_rate, epochs):
    weights = initial_weights
    
    for epoch in range(epochs):
        # Compute gradient
        gradient = compute_gradient(loss_function, weights)
        
        # Update weights (move opposite to gradient for minimization)
        weights = weights - learning_rate * gradient
        
    return weights
```

### 6.3 Example: Linear Regression with Multiple Features
**Model:** `y = w₁x₁ + w₂x₂ + b`

**Loss Function:** `L = (1/n) Σ(y_pred - y_actual)²`

**Gradients:**
```python
def compute_gradients(X, y, weights, bias):
    n = len(X)
    y_pred = X @ weights + bias
    error = y_pred - y
    
    # Partial derivatives
    dw1 = (2/n) * sum(error * X[:, 0])
    dw2 = (2/n) * sum(error * X[:, 1])
    db = (2/n) * sum(error)
    
    return [dw1, dw2], db
```

## 7. Higher-Order Partial Derivatives

### 7.1 Second Partial Derivatives
- `∂²f/∂x²` → second partial derivative w.r.t. x
- `∂²f/∂y²` → second partial derivative w.r.t. y
- `∂²f/∂x∂y` → mixed partial derivative

### 7.2 The Hessian Matrix
The Hessian contains all second partial derivatives:

For `f(x, y)`:
```
H = [∂²f/∂x²    ∂²f/∂x∂y]
    [∂²f/∂y∂x   ∂²f/∂y² ]
```

### 7.3 Applications in ML
- **Newton's Method**: Uses Hessian for faster optimization
- **Convexity Analysis**: Positive definite Hessian = convex function
- **Second-Order Optimization**: Algorithms like L-BFGS

## 8. Practical Examples

### 8.1 Logistic Regression
**Model:** `p = sigmoid(w₁x₁ + w₂x₂ + b)`

**Loss:** `L = -Σ[y log(p) + (1-y) log(1-p)]`

**Gradients:**
```python
def logistic_gradients(X, y, weights, bias):
    predictions = sigmoid(X @ weights + bias)
    error = predictions - y
    
    dw = X.T @ error / len(X)
    db = np.mean(error)
    
    return dw, db
```

### 8.2 Neural Network (Single Layer)
**Forward Pass:** `z = Wx + b`, `a = σ(z)`

**Backward Pass (Chain Rule):**
```python
# Output layer gradients
dL_da = compute_output_gradient(a, y)
da_dz = sigmoid_derivative(z)
dz_dW = x
dz_db = 1

# Final gradients
dL_dW = dL_da * da_dz * dz_dW
dL_db = dL_da * da_dz * dz_db
```

## 9. Computational Considerations

### 9.1 Automatic Differentiation
Modern frameworks (TensorFlow, PyTorch) compute gradients automatically:

```python
import torch

# Define function
x = torch.tensor([2.0, 3.0], requires_grad=True)
f = x[0]**2 + x[1]**2

# Compute gradient
f.backward()
print(x.grad)  # Output: [4.0, 6.0]
```

### 9.2 Numerical Gradients (for verification)
```python
def numerical_gradient(f, x, h=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    
    return grad
```

## 10. Common Pitfalls and Tips

### 10.1 Gradient Checking
Always verify your analytical gradients with numerical ones:
```python
analytical_grad = compute_analytical_gradient(weights)
numerical_grad = numerical_gradient(loss_function, weights)

# Should be very small (< 1e-7)
difference = np.linalg.norm(analytical_grad - numerical_grad)
```

### 10.2 Gradient Clipping
Prevent exploding gradients:
```python
def clip_gradients(gradients, max_norm=1.0):
    total_norm = np.linalg.norm(gradients)
    if total_norm > max_norm:
        gradients = gradients * (max_norm / total_norm)
    return gradients
```

### 10.3 Vanishing Gradients
- Use proper weight initialization
- Consider residual connections
- Use gradient-friendly activation functions

## 11. Directional Derivatives

Sometimes we want to know the rate of change in a specific direction (not necessarily coordinate directions).

**Directional Derivative:** `D_v f = ∇f · v`

Where `v` is a unit vector in the desired direction.

**Example:**
```python
# Gradient at point (2, 3)
gradient = np.array([4, 6])  # ∇f = [2x, 2y] at (2,3)

# Direction vector (normalized)
direction = np.array([1, 1]) / np.sqrt(2)

# Directional derivative
directional_deriv = np.dot(gradient, direction)
```

## 12. Applications Beyond Basic ML

### 12.1 Computer Vision
- **Image gradients** for edge detection
- **Optical flow** using spatial and temporal gradients

### 12.2 Natural Language Processing
- **Word embeddings** optimization
- **Attention mechanisms** gradient flow

### 12.3 Reinforcement Learning
- **Policy gradients** for strategy optimization
- **Value function** gradients

## 13. Conclusion

Partial derivatives and gradients are the workhorses of machine learning optimization. They tell us:

1. **Which direction** to move parameters
2. **How fast** the function changes
3. **Whether we're approaching** a minimum

Every time your model trains, it's computing thousands of partial derivatives and following gradients to find better parameters. Understanding this process gives you powerful insights into:

- Why certain optimizers work better
- How to debug training problems
- How to design better architectures

Next up: **The Chain Rule** - the mathematical foundation that makes backpropagation possible in deep neural networks! 