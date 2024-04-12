import numpy as np
import taichi as ti
import torch
from fluidlab.configs.macros import *
from pyquaternion import Quaternion

@ti.data_oriented
class VectorSensor:
    def __init__(self, AgentGameObject, device):
        self.AgentGameObject = AgentGameObject
        self.device = device
    def get_obs(self):
        state = self.AgentGameObject.effectors[0].get_state(self.AgentGameObject.sim.cur_substep_local)
        # print([state[0], state[1], 1 - state[2], -state[4], -state[5], state[6], state[3]])
        return torch.tensor([state[0], state[1], 1 - state[2], -state[4], -state[5], state[6], state[3]], dtype=torch.float32, device=self.device)
