import os
import gym
import numpy as np
from .fluid_env import FluidEnv
from yacs.config import CfgNode
from fluidlab.utils.misc import *
from fluidlab.configs.macros import *
from fluidlab.optimizer.policies import *
from fluidlab.fluidengine.taichi_env import TaichiEnv
from fluidlab.fluidengine.losses import *

class GatheringEasyEnv(FluidEnv):
    def __init__(self, version, loss=True, loss_type='diff', seed=None,  renderer_type='GGUI'):
        if seed is not None:
            self.seed(seed)

        self.horizon               = 840
        self.horizon_action        = 840
        self.target_file           = None
        self._n_obs_ptcls_per_body = 500
        self.loss                  = loss
        self.loss_type             = loss_type
        self.action_range          = np.array([-0.003, 0.003])
        self.renderer_type         = renderer_type
        self.reward                = None

        # create a taichi env
        self.taichi_env = TaichiEnv(
            dim=3,
            particle_density=1e6,
            max_substeps_local=50,
            gravity=(0.0, -20.0, 0.0),
            horizon=self.horizon,
        )
        self.build_env()
        self.gym_misc()

    def setup_agent(self):
        agent_cfg = CfgNode(new_allowed=True)
        agent_cfg.merge_from_file(get_cfg_path('agent_gatheringeasy.yaml'))
        self.taichi_env.setup_agent(agent_cfg)
        self.agent = self.taichi_env.agent

    def setup_statics(self):
        self.taichi_env.add_static(
            file='tank.obj',
            pos=(0.5, 0.2, 0.5),
            euler=(0.0, 0.0, 0.0),
            scale=(1.0, 0.92, 0.92),
            material=BOTTLE,
            has_dynamics=True,
        )

    def setup_bodies(self):
        self.taichi_env.add_body(
            type='cube',
            lower=(0.05, 0.3, 0.17),
            upper=(0.95, 0.45, 0.83),
            material=WATER,
        )
        self.taichi_env.add_body(
            type='mesh',
            file='duck.obj',
            pos=(0.22, 0.5, 0.45),
            scale=(0.10, 0.10, 0.10),
            euler=(0, -75.0, 0.0),
            color=(1.0, 1.0, 0.3, 1.0),
            filling='grid',
            material=RIGID,
        )
        self.taichi_env.add_body(
            type='mesh',
            file='duck.obj',
            pos=(0.28, 0.5, 0.57),
            scale=(0.10, 0.10, 0.10),
            euler=(0, -95.0, 0.0),
            color=(1.0, 0.5, 0.5, 1.0),
            filling='grid',
            material=RIGID,
        )

    def setup_boundary(self):
        self.taichi_env.setup_boundary(
            type='cube',
            lower=(0.06, 0.3, 0.18),
            upper=(0.94, 0.95, 0.82),
        )

    def setup_renderer(self):
        if self.renderer_type == 'GGUI':
            self.taichi_env.setup_renderer(
                type='GGUI',
                # render_particle=True,
                camera_pos=(-0.15, 2.82, 2.5),
                camera_lookat=(0.5, 0.5, 0.5),
                fov=30,
                lights=[{'pos': (0.5, 1.5, 0.5), 'color': (0.5, 0.5, 0.5)},
                        {'pos': (0.5, 1.5, 1.5), 'color': (0.5, 0.5, 0.5)}],
            )
        else:
            self.taichi_env.setup_renderer(
                type='GL',
                # render_particle=True,
                camera_pos=(-0.15, 2.82, 2.5),
                camera_lookat=(0.5, 0.5, 0.5),
                fov=30,
                light_pos=(0.5, 5.0, 0.55),
                light_lookat=(0.5, 0.5, 0.49),
            )

    def setup_loss(self):
        self.taichi_env.setup_loss(
            loss_cls=GatheringEasyLoss,
            type=self.loss_type,
            matching_mat=RIGID,
            weights={'dist': 1.0}
        )

    def trainable_policy(self, optim_cfg, init_range):
        return TorchGatheringPolicy(optim_cfg, init_range, self.agent.action_dim, self.horizon_action, self.action_range)
