import numpy as np
import taichi as ti
import torch
from fluidlab.configs.macros import *
from pyquaternion import Quaternion

@ti.data_oriented
class GridSensor:
    def __init__(self):
        ...
    def UpdateSensor(self, **kwargs):
        return NotImplementedError