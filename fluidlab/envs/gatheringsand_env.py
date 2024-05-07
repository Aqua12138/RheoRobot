import os
import gym
import numpy as np
import torch

from .fluid_env import FluidEnv
from yacs.config import CfgNode
from fluidlab.utils.misc import *
from fluidlab.configs.macros import *
from fluidlab.optimizer.policies import *
from fluidlab.fluidengine.taichi_env import TaichiEnv
from fluidlab.fluidengine.losses import *
from fluidlab.fluidengine.sensor import *
from fluidlab.fluidengine.rewards import *
from gym import spaces
class GatheringSandEnv(FluidEnv):
    def __init__(self, version, loss=True, loss_type='diff', seed=None,  renderer_type='GGUI', max_episode_steps=1000, stochastic_init=False, device="cpu", gamma=0.99):

        if seed is not None:
            self.seed(seed)
        else:
            self.seed(random.randint(1, 100))

        self.horizon               = max_episode_steps
        self.horizon_action        = max_episode_steps
        self.max_episode_steps        = max_episode_steps
        self.target_file           = None
        self._n_obs_ptcls_per_body = 500
        self.loss                  = loss
        self.loss_type             = loss_type
        self.action_range          = np.array([-0.007, 0.007])
        self.renderer_type         = renderer_type
        self.stochastic_init       = stochastic_init
        self.device                = device
        self.reward                = True
        self.gamma                 = gamma

        # create a taichi env
        self.taichi_env = TaichiEnv(
            dim=3,
            particle_density=1e6,
            max_substeps_local=200,
            gravity=(0.0, -9.8, 0.0),
            horizon=self.horizon,
        )
        self.build_env()
        self.gym_misc()
    def gym_misc(self):
        self.observation_space = spaces.Dict({
            'gridsensor3': spaces.Box(low=0, high=1, shape=(180, 90, 1), dtype=np.float32),
            'vector_obs': spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32),
            # 更多传感器可以继续添加
        })
        self.action_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)

    def setup_agent(self):
        agent_cfg = CfgNode(new_allowed=True)
        agent_cfg.merge_from_file(get_cfg_path('agent_gatheringsand.yaml'))
        self.taichi_env.setup_agent(agent_cfg)
        self.agent = self.taichi_env.agent

    def setup_statics(self):
        ...

    def setup_bodies(self):
        # self.taichi_env.add_body(
        #     type='cube',
        #     lower=(0.55, 0.3, 0.45),
        #     upper=(0.56, 0.31, 0.46),
        #     material=WATER,
        # )
        self.taichi_env.add_body(
            type='mesh',
            file='duck.obj',
            pos=(0.5, 0.5, 0.5),
            scale=(0.10, 0.10, 0.10),
            euler=(0, -75.0, 0.0),
            color=(1.0, 1.0, 0.3, 1.0),
            filling='grid',
            material=RIGID,
        )
        # self.taichi_env.add_body(
        #     type='mesh',
        #     file='duck.obj',
        #     pos=(0.28, 0.5, 0.57),
        #     scale=(0.10, 0.10, 0.10),
        #     euler=(0, -95.0, 0.0),
        #     color=(1.0, 0.5, 0.5, 1.0),
        #     filling='grid',
        #     material=RIGID,
        # )


    def setup_boundary(self):
        self.taichi_env.setup_boundary(
            type='cube',
            lower=(0.06, 0.25, 0.06),
            upper=(0.94, 0.95, 0.94),
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

    def setup_reward(self):
        self.taichi_env.setup_reward(
            reward_cls=GatheringEasyReward,
            type=self.loss_type,
            matching_mat=WATER,
            weights={'dist': 100},
            gamma=self.gamma
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
                            "MaxDistance": 1,
                            "MinDistance": 0,
                            "DistanceNormalization": 1,
                            "ObservationStacks": 1,
                            "device": self.device,
                            "n_particles": self.taichi_env.simulator.n_particles}
        vector_cfg = {"device": self.device}

        self.agent.add_sensor(sensor_handle=GridSensor3DGrad, sensor_cfg=gridsensor3d_cfg)
        # self.agent.add_sensor(sensor_handle=GridSensor2D, sensor_cfg=gridsensor2d_cfg)
        self.agent.add_sensor(sensor_handle=VectorSensor, sensor_cfg=vector_cfg)

    def trainable_policy(self, optim_cfg, init_range):
        return TorchGatheringPolicy(optim_cfg, init_range, self.agent.action_dim, self.horizon_action, self.action_range)

    def get_sensor_obs(self):
        obs = self.agent.get_obs()
        return {
            'gridsensor3': obs[0],
            'vector_obs': obs[1]
        }

    def reset(self):
        if self.stochastic_init:
            self.agent.set_target()
            # randomize the init state
            init_state = self._init_state

            random_particle_pos = np.random.uniform((0.1, 0.3, 0.1), (0.9, 0.3, 0.9))
            delta_pos = random_particle_pos - np.array([0.5, 0.5, 0.5])

            init_state['state']['x'] = self.x + delta_pos

            self.taichi_env.set_state(init_state['state'], grad_enabled=True)

        else:
            init_state = self._init_state
            self.taichi_env.set_state(init_state['state'], grad_enabled=True)
        self.taichi_env.reset_grad()
        return self.get_sensor_obs()

    def step(self, action: np.ndarray):
        action *= 0.35 * 2e-2
        action = np.clip(action, self.action_range[0], self.action_range[1])
        # print(action)

        self.taichi_env.step(action)

        obs = self.get_sensor_obs()
        reward = self._get_reward()

        self.render("human")

        assert self.t <= self.horizon
        if self.t == self.max_episode_steps:
            done = False
        else:
            done = False
        # print("t:", self.t)
        info = dict()
        return obs, torch.tensor(reward, dtype=torch.float32).to(self.device), torch.tensor(False, dtype=torch.bool).to(self.device), info

    def step_grad(self, action):
        action *= 0.35 * 2e-2
        action.clip(self.action_range[0], max=self.action_range[1])
        self.taichi_env.step_grad(action)

    def initialize_trajectory(self, s: int):
        self.taichi_env.set_state_anytime(self.sim_state, self.sim_substep_global, self.taichi_t)
        self.taichi_env.reset_grad()
        self.taichi_env.reset_step(int(s))
        return self.get_sensor_obs()

    def update_next_value(self, next_values):
        self.taichi_env.update_next_value(next_values)

    def update_gamma(self):
        self.taichi_env.update_gamma()

    def compute_actor_loss(self):
        self.taichi_env.compute_actor_loss()

    def compute_actor_loss_grad(self):
        self.taichi_env.compute_actor_loss_grad()

    def get_grad(self, m, n):
        # print(self.agent.get_grad(m, n))
        return self.agent.get_grad(m, n)
    def save_sim_state(self):
        self.sim_state = self.taichi_env.get_state()["state"]
        self.sim_substep_global = self.taichi_env.simulator.cur_substep_global
        self.taichi_t = self.taichi_env.t

    def set_next_state_grad(self, grad):
        self.taichi_env.set_next_state_grad(grad)


