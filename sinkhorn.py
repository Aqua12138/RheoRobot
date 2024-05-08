import taichi as ti
import math

ti.init(arch=ti.gpu)  # Use GPU for computation

# Parameters
eps = 0.1  # Regularization coefficient
max_iter = 100  # Maximum number of Sinkhorn iterations
p = 2  # Power for cost matrix computation
thresh = 1e-1  # Convergence threshold

# Declare fields
N = 2  # Assume maximum points in point clouds, adjust based on your dataset
D = 3  # Dimension of points
C = ti.field(dtype=ti.f32, shape=(N, N))
mu = ti.field(dtype=ti.f32, shape=N)
nu = ti.field(dtype=ti.f32, shape=N)
u = ti.field(dtype=ti.f32, shape=N)
v = ti.field(dtype=ti.f32, shape=N)
pi = ti.field(dtype=ti.f32, shape=(N, N))


@ti.func
def compute_cost_matrix(x, y):
    for i, j in ti.ndrange(N, N):
        diff = x[i] - y[j]  # Access the entire vector at points i and j
        C[i, j] = sum_elements(diff ** p)

@ti.func
def sum_elements(vec):
    total = 0.0
    for i in ti.static(range(vec.n)):
        total += vec[i]
    return total

@ti.kernel
def sinkhorn():
    # Reset fields
    for i in range(N):
        mu[i], nu[i], u[i], v[i] = 1.0 / N, 1.0 / N, 0.0, 0.0
    for i, j in C:
        C[i, j] = 0.0
        pi[i, j] = 0.0

    compute_cost_matrix(x_taichi, y_taichi)

    for _ in range(max_iter):
        for i in range(N):
            log_sum_exp = 0.0
            for j in range(N):
                M_ij = (-C[i, j] + u[i] + v[j]) / eps
                log_sum_exp += ti.exp(M_ij)
            u[i] = eps * (ti.log(mu[i]) - ti.log(log_sum_exp)) + u[i]

        for j in range(N):
            log_sum_exp = 0.0
            for i in range(N):
                M_ij = (-C[i, j] + u[i] + v[j]) / eps
                log_sum_exp += ti.exp(M_ij)
            v[j] = eps * (ti.log(nu[j]) - ti.log(log_sum_exp)) + v[j]

        # Early stopping condition could be added here

    for i, j in pi:
        M_ij = (-C[i, j] + u[i] + v[j]) / eps
        pi[i, j] = ti.exp(M_ij)


@ti.kernel
def compute_sinkhorn_distance() -> ti.f32:
    distance = 0.0
    for i, j in C:
        distance += pi[i, j] * C[i, j]
    return distance


# Example usage
# x = ti.Vector.field(D, dtype=ti.f32, shape=N)
# y = ti.Vector.field(D, dtype=ti.f32, shape=N)

# Populate x and y with your point cloud data
# 或 ti.gpu() 根据您的硬件选择

# 点云的维度和数量
import numpy as np

# Declare the sacred fields
x_taichi = ti.Vector.field(D, dtype=ti.f32, shape=N)
y_taichi = ti.Vector.field(D, dtype=ti.f32, shape=N)

# The enchanted function to transfer earthly data to the realm of Taichi
@ti.kernel
def transfer_data_to_taichi(x: ti.ext_arr(), y: ti.ext_arr()):
    for i in range(N):
        for d in ti.static(range(D)):
            x_taichi[i][d] = x[i, d]
            y_taichi[i][d] = y[i, d]


# The ritual to invoke the computation of the mystical distances
def perform_sinkhorn_distance_computation(x_np, y_np):
    transfer_data_to_taichi(x_np, y_np)
    sinkhorn()  # Assuming modifications to use x_taichi and y_taichi directly
    distance = compute_sinkhorn_distance()
    print(f"Sinkhorn Distance: {distance}")
    print(C)

# Sample data from the mortal world
a = np.array([[i, 0, 0] for i in range(2)])
b = np.array([[i, 1, 0] for i in range(2)])
# x_np = np.array([0, 0], [0, 1], [])
# y_np = np.array([1, 0], [1, 1])

# Invoke the higher powers
perform_sinkhorn_distance_computation(a, b)