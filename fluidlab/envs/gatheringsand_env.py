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
from fluidlab.fluidengine.sensor import *
class GatheringSandEnv(FluidEnv):
    def __init__(self, version, loss=True, loss_type='diff', seed=None,  renderer_type='GGUI'):

        if seed is not None:
            self.seed(seed)

        self.horizon               = 2000
        self.horizon_action        = 2000
        self.target_file           = None
        self._n_obs_ptcls_per_body = 500
        self.loss                  = loss
        self.loss_type             = loss_type
        self.action_range          = np.array([-0.003, 0.003])
        self.renderer_type         = renderer_type

        # create a taichi env
        self.taichi_env = TaichiEnv(
            dim=3,
            particle_density=1e6,
            max_substeps_local=50,
            gravity=(0.0, -9.8, 0.0),
            horizon=self.horizon,
        )
        self.build_env()
        self.gym_misc()

    def setup_agent(self):
        agent_cfg = CfgNode(new_allowed=True)
        agent_cfg.merge_from_file(get_cfg_path('agent_gatheringsand.yaml'))
        self.taichi_env.setup_agent(agent_cfg)
        self.agent = self.taichi_env.agent

    def setup_statics(self):
        self.taichi_env.add_static(
            file='single_wall.obj',
            pos=(0.5, 0.5, 0.5),
            euler=(0.0, 0.0, 0.0),
            scale=(1, 1, 1),
            material=TANK,
            has_dynamics=True,
        )

    def setup_bodies(self):
        self.taichi_env.add_body(
            type='cube',
            lower=(0.15, 0.6, 0.35),
            upper=(0.35, 0.65, 0.65),
            material=WATER,
        )


    def setup_boundary(self):
        # do not setup boundary bigger than 1.0 or smaller than 0.0
        self.taichi_env.setup_boundary(
            type='cube',
            lower=(0.05, 0.05, 0.05),
            upper=(0.95, 0.95, 0.95),
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
            matching_mat=WATER,
            weights={'dist': 0.1}
        )

    def setup_sensors(self):
        # setup sensor and build
        gridsensor2d_cfg = {"SensorName": "cup_gridsensor2d",
                            "CellScale": (0.05, 0.05),
                            "GridSize": (20, 20),
                            "RotateWithAgent": False,
                            "ObservationStacks": 1,
                            "GlobalPosition": [0.5, 0.5, 0.5]}

        gridsensor3d_cfg = {"SensorName": "cup_gridsensor3d",
                            "CellArc": 2,
                            "LatAngleNorth": 90,
                            "LatAngleSouth": 90,
                            "LonAngle": 180,
                            "MaxDistance": 0.5,
                            "MinDistance": 0,
                            "DistanceNormalization": 1,
                            "ObservationStacks": 1}

        self.agent.add_sensor(sensor_handle=GridSensor3D, sensor_cfg=gridsensor3d_cfg)
        # self.agent.add_sensor(sensor_handle=GridSensor2D, sensor_cfg=gridsensor2d_cfg)
        self.agent.add_sensor(sensor_handle=VectorSensor)

    def trainable_policy(self, optim_cfg, init_range):
        return TorchGatheringPolicy(optim_cfg, init_range, self.agent.action_dim, self.horizon_action, self.action_range)

    # def get_obs(self):
    #     obs = self.agent.get_obs()
    #     return obs
