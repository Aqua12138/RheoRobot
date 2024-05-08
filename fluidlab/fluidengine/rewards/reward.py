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


@ti.data_oriented
class Reward:
    def __init__(self, max_loss_steps, weights=None, target_file=None, gamma=0.99):
        self.weights = weights
        self.target_file = target_file
        self.inf = 1e8
        self.max_loss_steps = max_loss_steps
        self._gamma = gamma

        self.rew = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)
        self.rew_acc = ti.field(dtype=DTYPE_TI, shape=(33,), needs_grad=True)
        self.actor_loss = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)
        self.next_values = ti.field(dtype=DTYPE_TI, shape=(33,), needs_grad=False)
        self.gamma = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=False)
        self.s = 0


    def build(self, sim):
        self.sim    = sim
        self.res    = sim.res
        self.n_grid = sim.n_grid
        self.dx     = sim.dx
        self.dim    = sim.dim
        
        if self.sim.agent is not None:
            self.agent = sim.agent

        if self.sim.particles is not None:
            self.particle_x    = sim.particles.x
            self.particle_mat  = sim.particles_i.mat
            self.particle_used = sim.particles_ng.used
            self.n_particles   = sim.n_particles
            self.grid_mass     = sim.grid.mass

        if self.sim.smoke_field is not None:
            self.smoke_field = sim.smoke_field

        if self.target_file is not None:
            self.load_target(self.target_file)

        self.reset()

    def reset_grad(self):
        self.rew_acc.grad.fill(0)
        self.actor_loss.grad.fill(1)
        self.rew.grad.fill(0)
        
    def load_target(self, path):
        pass

    @ti.kernel
    def clear_loss(self):
        self.rew_acc.fill(0)
        self.rew_acc.grad.fill(0)

        self.actor_loss.fill(0)
        self.actor_loss.grad.fill(1.0/64.0)

        self.rew.fill(0)
        self.rew.grad.fill(0)

        self.gamma.fill(1)

        self.next_values.fill(0)

    @ti.kernel
    def clear_losses(self):
        pass

    def reset(self):
        self.clear_loss()
        self.clear_losses()

    def step(self):
        # compute loss for step self.sim.cur_step_global-1
        self.s = self.s+1
        self.compute_step_reward(self.s-1, self.sim.cur_substep_local)
        # print("step:", self.sim.cur_step_global-1)

    def step_grad(self):
        # compute loss for step self.sim.cur_step_global-1
        self.compute_step_reward_grad(self.s-1, self.sim.cur_substep_local)
        self.s = self.s-1
        # print("step_grad:", self.sim.cur_step_global - 1)

    def step_next_values(self, next_value):
        self.next_values[self.sim.cur_step_global] = next_value

    def update_gamma(self):
        self.gamma[None] = self.gamma[None] * self._gamma







