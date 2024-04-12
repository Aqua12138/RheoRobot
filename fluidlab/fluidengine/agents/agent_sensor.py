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
        self.sim.update_gridSensor(reset=False)
        for sensor in self.sensors:
            sensor_obs.append(sensor.get_obs())
            # sensor_obs.append(np.asarray(group_obs, dtype=np.float32))

        return sensor_obs

    def add_sensor(self, sensor_handle, sensor_cfg=None):
        sensor = sensor_handle(**sensor_cfg, AgentGameObject=self)
        self.sensors.append(sensor)



