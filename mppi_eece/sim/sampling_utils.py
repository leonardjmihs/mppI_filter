import numpy as np

def approximate_gradient_sampling(f, x, num_samples=100, epsilon=1e-4, random_state=None):
    """
    Approximates the gradient of a scalar function f at point x using random sampling.
    Args:
        f: function mapping np.ndarray -> float
        x: np.ndarray, input point (arbitrary shape)
        num_samples: number of random directions to sample
        epsilon: step size for finite differences
        random_state: optional np.random.RandomState or int seed
    Returns:
        grad_est: np.ndarray, same shape as x, estimated gradient
    """
    x = np.asarray(x)
    grad_shape = x.shape
    grad_est = np.zeros_like(x, dtype=float)
    rng = np.random.default_rng(random_state)
    for _ in range(num_samples):
        v = rng.normal(size=grad_shape)
        v /= np.linalg.norm(v) + 1e-12  # normalize direction
        f_plus = f(x + epsilon * v)
        f_minus = f(x - epsilon * v)
        directional_deriv = (f_plus - f_minus) / (2 * epsilon)
        grad_est += directional_deriv * v
    grad_est /= num_samples
    return grad_est

# Example usage:
# def my_func(x):
#     return np.sum(x ** 2)
# x0 = np.array([1.0, 2.0, 3.0])
# grad = approximate_gradient_sampling(my_func, x0)
# print(grad)  # Should be close to 2*x0
