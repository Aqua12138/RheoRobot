import os
import torch
import numpy as np
import taichi as ti
import pickle as pkl
from sklearn.neighbors import KDTree
from fluidlab.fluidengine.simulators import MPMSimulator
from fluidlab.configs.macros import *
from fluidlab.utils.misc import *
import matplotlib.pyplot as plt
from .reward import Reward

@ti.data_oriented
class PouringReward(Reward):
    def __init__(
            self,
            type,
            matching_mat,
            **kwargs,
        ):
        super().__init__(**kwargs)

        self.matching_mat = matching_mat

        self.eps = 0.1  # Regularization coefficient
        self.max_iter = 1  # Maximum number of Sinkhorn iterations
        self.p = 2  # Power for cost matrix computation
        self.thresh = 1e-1  # Convergence threshold

    def build(self, sim):
        self.dist_weight = self.weights['dist']
        self.dist_reward = ti.field(dtype=DTYPE_TI, shape=(self.max_loss_steps+1,), needs_grad=True)

        super().build(sim)

        self.N = self.sim.n_particles # 根据matching_mat调整
        self.D = 3  # Dimension of points
        self.C = ti.field(dtype=ti.f32, shape=(33, self.N, self.N), needs_grad=True)
        self.mu = ti.field(dtype=ti.f32, shape=self.N)
        self.nu = ti.field(dtype=ti.f32, shape=self.N)
        self.u = ti.field(dtype=ti.f32, shape=(33, self.N), needs_grad=True)
        self.v = ti.field(dtype=ti.f32, shape=(33, self.N), needs_grad=True)
        self.pi = ti.field(dtype=ti.f32, shape=(33, self.N, self.N), needs_grad=True)

        self.target = ti.Vector.field(self.dim, dtype=DTYPE_TI, shape=self.N, needs_grad=False)
        self.initialize_positions()

    @ti.kernel
    def initialize_positions(self):
        for p in range(self.N):
            self.target[p][0] = ti.random() * (0.6 - 0.4) + 0.4  # x component in [0.4, 0.6]
            self.target[p][1] = ti.random() * (0.2 - 0.0) + 0.0  # y component in [0, 0.2]
            self.target[p][2] = ti.random() * (0.6 - 0.4) + 0.4  # z component in [0.4, 0.6]

    def reset_grad(self):
        super().reset_grad()
        self.dist_reward.grad.fill(0)
        
    @ti.kernel
    def clear_losses(self):
        self.dist_reward.fill(0)
        self.dist_reward.grad.fill(0)
        self.init_dist_reward_kernel()

    @ti.kernel
    def clear_gradients(self):
        for i, j in self.C:
            self.C.grad[i, j] = 0.0
            self.pi.grad[i, j] = 0.0
        for i in range(self.N):
            self.u.grad[i] = 0.0
            self.v.grad[i] = 0.0
    def compute_step_reward(self, s, f):
        self.clear_gradients()

        self.init_sinkhorn_kernel()
        self.compute_cost_matrix(f)
        self.update_potentials_kernel()
        self.compute_transport_plan_kernel()
        self.compute_sinkhorn_distance(s)
        self.sum_up_reward_kernel(s)
        self.compute_reward_kernel(s)

    def compute_step_reward_grad(self, s, f):
        self.compute_reward_kernel.grad(s)
        self.sum_up_reward_kernel.grad(s)
        self.compute_sinkhorn_distance.grad(s)
        self.compute_transport_plan_kernel.grad()
        self.update_potentials_kernel.grad()
        self.compute_cost_matrix.grad(f)
        self.init_sinkhorn_kernel.grad()
        print("compute")
        # self.debug_grad(f)


    def compute_actor_loss(self):
        self.compute_actor_loss_kernel(self.sim.cur_step_global-1)

    def compute_actor_loss_grad(self):
        self.compute_actor_loss_kernel.grad(self.sim.cur_step_global-1)

    @ti.func
    def sum_elements(self, vec):
        total = 0.0
        for i in ti.static(range(vec.n)):
            total += vec[i]
        return total
    @ti.kernel
    def compute_cost_matrix(self, f: ti.i32):
        for i, j in ti.ndrange(self.N, self.N):
            diff = self.particle_x[f, i] - self.target[j]
            self.C[i, j] = self.sum_elements(diff ** self.p)
    # sinkhorn distance
    @ti.kernel
    def init_sinkhorn_kernel(self):
        # Reset fields
        for i in range(self.N):
            self.mu[i], self.nu[i], self.u[i], self.v[i] = 1.0 / self.N, 1.0 / self.N, 0.0, 0.0
        for i, j in self.C:
            self.C[i, j] = 0.0
            self.pi[i, j] = 0.0

    @ti.kernel
    def update_potentials_kernel(self):
        for _ in range(self.max_iter):
            for i in range(self.N):
                log_sum_exp = 0.0
                for j in range(self.N):
                    M_ij = (-self.C[i, j] + self.u[i] + self.v[j]) / self.eps
                    log_sum_exp += ti.exp(M_ij)
                self.u[i] = self.eps * (ti.log(self.mu[i]) - ti.log(log_sum_exp)) + self.u[i]

            for j in range(self.N):
                log_sum_exp = 0.0
                for i in range(self.N):
                    M_ij = (-self.C[i, j] + self.u[i] + self.v[j]) / self.eps
                    log_sum_exp += ti.exp(M_ij)
                self.v[j] = self.eps * (ti.log(self.nu[j]) - ti.log(log_sum_exp)) + self.v[j]
    @ti.kernel
    def compute_transport_plan_kernel(self):
        for i, j in self.pi:
            M_ij = (-self.C[i, j] + self.u[i] + self.v[j]) / self.eps
            self.pi[i, j] = ti.exp(M_ij)


    @ti.kernel
    def compute_sinkhorn_distance(self, s: ti.i32):
        for i, j in self.C:
            self.dist_reward[s+1] += self.pi[i, j] * self.C[i, j]
            # print(self.dist_reward[s+1])

    @ti.func
    def init_dist_reward_kernel(self):
        for p in range(self.n_particles):
            if self.particle_used[0, p] and self.particle_mat[p] == self.matching_mat:
                if self.particle_x[0, p][0] > 0.4 and self.particle_x[0, p][0] < 0.6 and \
                    self.particle_x[0, p][1] > 0 and self.particle_x[0, p][1] < 0.3 and \
                    self.particle_x[0, p][2] > 0.4 and self.particle_x[0, p][2] < 0.6:
                    self.dist_reward[0] += 1

    @ti.kernel
    def sum_up_reward_kernel(self, s: ti.i32):
        self.rew[None] = ((self.dist_reward[s] * self.dist_weight) - (self.dist_reward[s+1] * self.dist_weight))
        if ti.abs(self.rew[None] / self.dist_weight) < 1e-5:
            self.rew[None] = 0
        # print(self.rew[None])
    @ti.kernel
    def compute_reward_kernel(self, s: ti.i32):
        self.rew_acc[s+1] = self.rew_acc[s] + self.gamma[None] * self.rew[None]
    @ti.kernel
    def compute_total_loss_kernel(self, s_start: ti.i32, s_end: ti.i32):
        for s in range(s_start, s_end):
            self.total_loss[None] += self.step_loss[s]
    @ti.kernel
    def compute_actor_loss_kernel(self, s: ti.i32):
        self.actor_loss[None] = -self.rew_acc[s+1]

    def get_final_loss_grad(self):
        self.compute_total_loss_kernel.grad(self.temporal_range[0], self.temporal_range[1])


    # @ti.kernel
    # def debug_grad(self, f: ti.i32):
    #     print("particle_x_grad", self.dist_reward[1].grad[f, 0][0])

    def expand_temporal_range(self):
        if self.temporal_range_type == 'expand':
            loss_improved = (self.best_loss - self.total_loss[None])
            loss_improved_rate = loss_improved / self.best_loss
            if loss_improved_rate < self.plateau_thresh[0] or loss_improved < self.plateau_thresh[1]:
                self.plateau_count += 1
                print('Plateaued!!!', self.plateau_count)
            else:
                self.plateau_count = 0

            if self.best_loss > self.total_loss[None]:
                self.best_loss = self.total_loss[None]

            if self.plateau_count >= self.plateau_count_limit:
                self.plateau_count = 0
                self.best_loss = self.inf

                self.temporal_range[1] = min(self.max_loss_steps, self.temporal_range[1] + self.temporal_expand_speed)
                print(f'temporal range expanded to {self.temporal_range}')
            
    def get_step_reward(self):
        cur_step_reward = self.rew[None]
        reward = cur_step_reward

        reward_info = {}
        reward_info['reward'] = reward
        return reward_info

    def get_final_reward(self):
        self.compute_total_loss_kernel(self.temporal_range[0], self.temporal_range[1])
        self.expand_temporal_range()
        
        loss_info = {
            'loss': self.total_loss[None],
            'last_step_loss': self.step_loss[self.max_loss_steps-1],
            'temporal_range': self.temporal_range[1],
            'reward': np.sum((150 - self.step_loss.to_numpy()) * 0.01)
        }

        return loss_info
