import ray
import taichi as ti

ray.init()

@ray.remote
def distributed_simulation(x, y):
    ti.init(arch=ti.gpu)  # 每次调用时初始化 Taichi

    @ti.kernel
    def compute_force(x: ti.f32, y: ti.f32) -> ti.f32:
        return x * y * 0.5

    def run_simulation(x, y):
        result = compute_force(x, y)
        return result

    return run_simulation(x, y)

# 创建一些任务
x_values = [1.0, 2.0, 3.0, 4.0]
y_values = [5.0, 6.0, 7.0, 8.0]
futures = [distributed_simulation.remote(x, y) for x, y in zip(x_values, y_values)]

# 收集结果
results = ray.get(futures)
print("Simulation results:", results)
