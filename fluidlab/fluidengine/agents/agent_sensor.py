import yaml
import taichi as ti
import numpy as np
from .agent import Agent
from fluidlab.fluidengine.effectors import *


@ti.data_oriented
class AgentSensor(Agent):
    # Agent with one Rigid
    def __init__(self, **kwargs):
        super(AgentSensor, self).__init__(**kwargs)
    def build(self, sim):
        super(AgentSensor, self).build(sim)

        assert self.n_effectors == 1
        assert isinstance(self.effectors[0], Rigid)
        self.rigid = self.effectors[0]
        self.sensors = []
    @ti.func
    def collide(self, f, pos_world, mat_v, dt):
        return self.rigid.collide(f, pos_world, mat_v, dt)

    def get_obs(self):
        sensor_obs = []
        for sensor in self.sensors:
            sensor_obs.append(sensor.get_obs())
            # sensor_obs.append(np.asarray(group_obs, dtype=np.float32))

        return sensor_obs

    def add_sensor(self, sensor_handle, sensor_cfg=None):
        sensor = sensor_handle(**sensor_cfg, AgentGameObject=self, sim=self.sim)
        self.sensors.append(sensor)

    def set_target(self):
        self.target = np.random.uniform(low=(0.05, 0.05, 0.05), high=(0.95, 0.95, 0.95))
        self.effectors[0].set_target(self.target)
        # print("target:", self.target)

    def set_next_state_grad(self, grad):
        self.effectors[0].set_next_state_grad(self.sim.cur_substep_local, grad["vector_obs"])
        self.sensors[0].set_next_state_grad(self.sim.cur_step_global, grad["grid_sensor3"])

    def reset_grad(self):
        for i in range(self.n_effectors):
            self.effectors[i].reset_grad()
        self.sensors[0].reset_grad()

    def set_state(self, f, state):
        for i in range(self.n_effectors):
            self.effectors[i].set_state(f, state[i])
        self.sensors[0].reset()



