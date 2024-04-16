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
class GatheringEasyReward(Reward):
    def __init__(
            self,
            type,
            matching_mat,
            **kwargs,
        ):
        super().__init__(**kwargs)

        self.matching_mat = matching_mat
        
        if type == 'diff':
            self.plateau_count_limit     = 10
            self.temporal_expand_speed   = 120
            self.temporal_init_range_end = 120
            self.temporal_range_type     = 'expand'
            self.plateau_thresh          = [1e-6, 0.1]
        elif type == 'default':
            self.temporal_range_type     = 'all'
        else:
            assert False


    def build(self, sim):
        self.dist_weight = self.weights['dist']
        self.dist_reward = ti.field(dtype=DTYPE_TI, shape=(self.max_loss_steps,), needs_grad=True)

        if self.temporal_range_type == 'last':
            self.temporal_range = [self.max_loss_steps-1, self.max_loss_steps]
        elif self.temporal_range_type == 'all':
            self.temporal_range = [0, self.max_loss_steps]
        elif self.temporal_range_type == 'expand':
            self.temporal_range = [0, self.temporal_init_range_end]
            self.best_loss = self.inf
            self.plateau_count = 0

        super().build(sim)

    def reset_grad(self):
        super().reset_grad()
        self.dist_reward.grad.fill(0)
        
    @ti.kernel
    def clear_losses(self):
        self.dist_reward.fill(0)
        self.dist_reward.grad.fill(0)

    def compute_step_reward(self, s, f):
        self.compute_dist_reward_kernel(s, f)
        self.sum_up_reward_kernel(s)
        self.compute_reward_kernel(s)

    def compute_step_reward_grad(self, s, f):
        self.compute_reward_kernel.grad(s)
        self.sum_up_reward_kernel.grad(s)
        self.compute_dist_reward_grad(s, f)

    def compute_dist_reward(self, s, f):
        self.compute_dist_reward_kernel(s, f)


    def compute_dist_reward_grad(self, s, f):
        self.compute_dist_reward_kernel.grad(s, f)

    def compute_actor_loss(self):
        self.compute_actor_loss_kernel(self.sim.cur_step_global-1)

    def compute_actor_loss_grad(self):
        self.compute_actor_loss_kernel.grad(self.sim.cur_step_global-1)

    @ti.kernel
    def compute_dist_reward_kernel(self, s: ti.i32, f: ti.i32):
        for p in range(self.n_particles):
            if self.particle_used[f, p] and self.particle_mat[p] == self.matching_mat:
                self.dist_reward[s] += ti.abs(self.particle_x[f, p][0] - 0.8)


    @ti.kernel
    def sum_up_reward_kernel(self, s: ti.i32):
        self.rew[None] = self.dist_reward[s] * self.dist_weight * 0.01
        print("taichi:", self.rew[None])

    @ti.kernel
    def compute_reward_kernel(self, s: ti.i32):
        self.rew_acc[s+1] = self.rew_acc[s] + self.gamma[None] * self.rew[None]

    @ti.kernel
    def compute_total_loss_kernel(self, s_start: ti.i32, s_end: ti.i32):
        for s in range(s_start, s_end):
            self.total_loss[None] += self.step_loss[s]
    @ti.kernel
    def compute_actor_loss_kernel(self, s: ti.i32):
        self.actor_loss[None] = -self.rew_acc[s+1] - self.gamma[None] * self._gamma * self.next_values[s+1]
        print("actor_loss:", self.actor_loss[None], "rew_acc:", self.rew_acc[s+1], "gamma:", self.gamma[None], "next_value:", self.next_values[s+1])

    def get_final_loss_grad(self):
        self.compute_total_loss_kernel.grad(self.temporal_range[0], self.temporal_range[1])
        # self.debug_grad(self.temporal_range[0], self.temporal_range[1])

    @ti.kernel
    def debug_grad(self, s_start: ti.i32, s_end: ti.i32):
        for s in range(s_start, s_end):
            print("step loss grad:", self.step_loss.grad[s])
    #
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
        reward =  cur_step_reward

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
