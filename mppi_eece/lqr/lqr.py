import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 200
dt = 1
n = 4
m = 2

# System matrices
A = np.block([[np.eye(2), np.eye(2)],
              [np.zeros((2, 2)), np.eye(2)]])
B = np.block([[np.zeros((2, 2))],
              [np.eye(2)]])

# Cost matrices
Q = np.block([[np.eye(2), np.zeros((2, 2))],
              [np.zeros((2, 2)), np.zeros((2, 2))]])
Qf = Q

# Reference trajectory
t = np.arange(N)
z = np.array([np.cos(2 * np.pi * t / 100),
              np.sin(2 * np.pi * t / 100)])

# Initial state
x0 = np.array([1, 0, 0, 0])

# mu values to test
mu_values = [1, 10, 100, 1000]
colors = plt.cm.tab10(np.linspace(0, 1, len(mu_values)))

# Create figure
plt.figure(figsize=(8, 8))
plt.grid(True)
plt.title('Hovercraft LQR Tracking for different μ')
plt.xlabel('x')
plt.ylabel('y')

# Loop over mu values
for idx, mu in enumerate(mu_values):
    R = mu * np.eye(2)
    
    # Initialize P, q, K
    P = [None] * (N + 1)
    q = [None] * (N + 1)
    K = [None] * N
    
    # Terminal conditions
    P[N] = Qf
    q[N] = -Qf @ np.concatenate([z[:, -1], np.zeros(2)])
    
    # Backward pass
    for t_idx in range(N - 1, -1, -1):
        K[t_idx] = -np.linalg.solve(R + B.T @ P[t_idx + 1] @ B, 
                                     B.T @ P[t_idx + 1] @ A)
        Acl = A + B @ K[t_idx]
        P[t_idx] = Q + Acl.T @ P[t_idx + 1] @ A
        zt = np.concatenate([z[:, t_idx], np.zeros(2)])
        q[t_idx] = Acl.T @ q[t_idx + 1] - Q @ zt
    
    # Forward pass
    x = np.zeros((n, N))
    uol = np.zeros((m, N))
    x[:, 0] = x0
    
    for t_idx in range(N - 1):
        uol[:, t_idx] = K[t_idx] @ x[:, t_idx] - \
            np.linalg.solve(R + B.T @ P[t_idx + 1] @ B, B.T @ q[t_idx + 1])
        x[:, t_idx + 1] = A @ x[:, t_idx] + B @ uol[:, t_idx]
    
    # Plot trajectory
    plt.plot(x[0, :], x[1, :], color=colors[idx], linewidth=1.5, 
             label=f'μ={mu}')

plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.show()