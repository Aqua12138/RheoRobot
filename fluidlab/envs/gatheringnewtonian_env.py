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
class GatheringNewtonianEnv(FluidEnv):
    def __init__(self, version, loss=True, loss_type='diff', seed=None,  renderer_type='GGUI', max_episode_steps=1000, stochastic_init=False, device="cpu", gamma=0.99):

        if seed is not None:
            self.seed(seed)
        else:
            ...

        self.horizon               = max_episode_steps
        self.horizon_action        = max_episode_steps
        self.max_episode_steps     = max_episode_steps
        self.target_file           = None
        self._n_obs_ptcls_per_body = 500
        self.loss                  = loss
        self.loss_type             = loss_type
        self.action_range          = np.array([[-0.007, -0.007, -0.007],
                                              [0.007, 0.007, 0.007]])
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
        self.action_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
    def setup_agent(self):
        agent_cfg = CfgNode(new_allowed=True)
        agent_cfg.merge_from_file(get_cfg_path('agent_gatheringnewtonian.yaml'))
        self.taichi_env.setup_agent(agent_cfg)
        self.agent = self.taichi_env.agent
    def setup_statics(self):
        self.taichi_env.add_static(
            file='table.obj',
            pos=(0.25, 0.5, 0.5),
            euler=(0.0, 0.0, 0.0),
            scale=(0.5, 1, 1),
            material=TANK,
            has_dynamics=True,
        )
    def setup_bodies(self):
        self.taichi_env.add_body(
            type='cylinder',
            center=(0.25, 0.6, 0.5),
            height=0.05,
            radius=0.03,
            material=MILK_VIS,
        )
    def setup_boundary(self):
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

    def setup_reward(self):
        self.taichi_env.setup_reward(
            reward_cls=PouringReward,
            type=self.loss_type,
            matching_mat=WATER,
            weights={'dist': 1000},
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
            # randomize the init state
            init_state = self._init_state

            random_particle_pos = np.random.uniform((0.1, 0.6, 0.1), (0.4, 0.6, 0.9))
            random_agent_pos = np.random.uniform((0.1, 0.65, 0.1), (0.9, 0.9, 0.9))
            delta_pos = random_particle_pos - np.array([0.25, 0.6, 0.5])

            init_state['state']['x'] = self.x + delta_pos
            init_state['state']['agent'][0][0:3] = random_agent_pos

            self.taichi_env.set_state(init_state['state'], grad_enabled=True)

        else:
            init_state = self._init_state
            self.taichi_env.set_state(init_state['state'], grad_enabled=True)
        self.agent.sensors[0].reset()
        self.taichi_env.reward.reset()
        self.taichi_env.reset_grad()
        return self.get_sensor_obs()

    def step(self, action: np.ndarray):
        action *= 0.35 * 2e-2
        action = np.clip(action, self.action_range[0], self.action_range[1])
        # print(action)

        self.taichi_env.step(action)

        obs = self.get_sensor_obs()
        reward = self._get_reward()
        # Define the field
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
        # reset sensor, sensor grad
        self.taichi_env.set_state_anytime(self.sim_state, self.sim_substep_global, self.taichi_t)
        self.agent.sensors[0].reset()
        self.taichi_env.reward.reset()
        # reset sensor grad, reward grad(self.rew_acc[s], self.gamma.fill(1.0), self.actor_loss.fill(0.0), dist)

        self.taichi_env.reset_grad()
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

    def demo_policy(self, user_input=False):
        if user_input:
            # init_p = self.agents_state
            return KeyboardPolicy_vxy(v_lin=0.007, v_ang=0.021)
        else:
            comp_actions_p = np.zeros((1, self.agent.action_dim))
            comp_actions_v = np.zeros((self.horizon_action, self.agent.action_dim))
            init_p = np.array([0.15, 0.65, 0.5])
            x_range = 0.7
            current_p = np.array(init_p)
            amp_range = np.array([0.15, 0.25])
            for i in range(self.horizon_action):
                target_i = i + 1
                target_x = init_p[0] + target_i/self.horizon_action*x_range
                target_y = init_p[1]
                cycles = 3
                target_rad = target_i/self.horizon_action*(np.pi*2)*cycles
                target_amp = amp_range[1] - np.abs((target_i*2/self.horizon_action) - 1) * (amp_range[1] - amp_range[0])
                target_z = np.sin(target_rad)*target_amp+0.5
                target_p = np.array([target_x, target_y, target_z])

                comp_actions_v[i] = target_p - current_p
                current_p += comp_actions_v[i]

            comp_actions_p[0] = init_p
            comp_actions = np.vstack([comp_actions_v, comp_actions_p])
            return ActionsPolicy(comp_actions)

