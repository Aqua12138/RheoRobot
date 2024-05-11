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
        self.dist_reward = ti.field(dtype=DTYPE_TI, shape=(33,), needs_grad=True)

        self.N = sim.n_particles # 根据matching_mat调整
        self.D = 3  # Dimension of points
        self.C = ti.field(dtype=ti.f32, shape=(33, self.N, self.N), needs_grad=True)
        self.mu = ti.field(dtype=ti.f32, shape=self.N)
        self.nu = ti.field(dtype=ti.f32, shape=self.N)
        self.u = ti.field(dtype=ti.f32, shape=(33, self.N), needs_grad=True)
        self.v = ti.field(dtype=ti.f32, shape=(33, self.N), needs_grad=True)
        self.pi = ti.field(dtype=ti.f32, shape=(33, self.N, self.N), needs_grad=True)

        self.target = ti.Vector.field(3, dtype=DTYPE_TI, shape=self.N, needs_grad=False)
        self.initialize_positions()

        super().build(sim)
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

        self.mu.fill(1/self.N)
        self.nu.fill(1/self.N)
        self.C.fill(0)
        self.u.fill(0)
        self.v.fill(0)
        self.pi.fill(0)

        self.C.grad.fill(0)
        self.u.grad.fill(0)
        self.v.grad.fill(0)
        self.pi.grad.fill(0)

    def set_reward(self, s):
        self.compute_cost_matrix(s, s)
        self.update_potentials_kernel(s)
        self.compute_transport_plan_kernel(s)
        self.init_compute_sinkhorn_distance()
        # self.sum_up_reward_kernel(s)
        # self.compute_reward_kernel(s)

    def compute_step_reward(self, s, f):
        self.compute_cost_matrix(s, f)
        self.update_potentials_kernel(s)
        self.compute_transport_plan_kernel(s)
        self.compute_sinkhorn_distance(s)
        self.sum_up_reward_kernel(s)
        self.compute_reward_kernel(s)

    def compute_step_reward_grad(self, s, f):
        self.compute_reward_kernel.grad(s)
        self.sum_up_reward_kernel.grad(s)
        self.compute_sinkhorn_distance.grad(s)
        self.compute_transport_plan_kernel.grad(s)
        self.update_potentials_kernel.grad(s)
        self.compute_cost_matrix.grad(s, f)
        # print("compute")
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
    def compute_cost_matrix(self, s: ti.i32, f: ti.i32):
        for i, j in ti.ndrange(self.N, self.N):
            diff = self.particle_x[f, i] - self.target[j]
            self.C[s, i, j] = self.sum_elements(diff ** self.p)
    # sinkhorn distance


    @ti.kernel
    def update_potentials_kernel(self, s: ti.i32):
        for _ in range(self.max_iter):
            for i in range(self.N):
                log_sum_exp = 0.0
                for j in range(self.N):
                    M_ij = (-self.C[s, i, j] + self.u[s, i] + self.v[s, j]) / self.eps
                    log_sum_exp += ti.exp(ti.max(M_ij, -20))
                self.u[s, i] = self.eps * (ti.log(ti.max(self.mu[i], 1e-7)) - ti.log(ti.max(log_sum_exp, 1e-7))) + self.u[s, i]

            for j in range(self.N):
                log_sum_exp = 0.0
                for i in range(self.N):
                    M_ij = (-self.C[s, i, j] + self.u[s, i] + self.v[s, j]) / self.eps
                    log_sum_exp += ti.exp(ti.max(M_ij, -20))
                self.v[s, j] = self.eps * (ti.log(self.nu[j]) - ti.log(ti.max(log_sum_exp, 1e-7))) + self.v[s, j]
    @ti.kernel
    def compute_transport_plan_kernel(self, s: ti.i32):
        for i, j in ti.ndrange(self.N, self.N):
            M_ij = (-self.C[s, i, j] + self.u[s, i] + self.v[s, j]) / self.eps
            self.pi[s, i, j] = ti.exp(M_ij)


    @ti.kernel
    def compute_sinkhorn_distance(self, s: ti.i32):
        for i, j in ti.ndrange(self.N, self.N):
            self.dist_reward[s+1] += self.pi[s, i, j] * self.C[s, i, j]

            # print(self.dist_reward[s+1])
    @ti.kernel
    def init_compute_sinkhorn_distance(self):
        for i, j in ti.ndrange(self.N, self.N):
            self.dist_reward[0] += self.pi[0, i, j] * self.C[0, i, j]
    @ti.kernel
    def sum_up_reward_kernel(self, s: ti.i32):
        self.rew[None] = ((self.dist_reward[s] * self.dist_weight) - (self.dist_reward[s+1] * self.dist_weight))
        if ti.abs(self.rew[None] / self.dist_weight) < 1e-5:
            self.rew[None] = 0
        # print("rew:", self.rew[None], "sinkhorn s: ", s, self.dist_reward[s], "sinkhorn s+1: ", s+1, self.dist_reward[s+1])
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
