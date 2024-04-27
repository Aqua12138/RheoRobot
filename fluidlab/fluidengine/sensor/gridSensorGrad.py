import numpy as np
import taichi as ti
import torch
from fluidlab.configs.macros import *
from pyquaternion import Quaternion
from .gridSensor import GridSensor
from fluidlab.utils.geom import quaternion_to_rotation_matrix

@ti.data_oriented
class GridSensor3DGrad(GridSensor):
    def __init__(self, SensorName, AgentGameObject, ObservationStacks, CellArc, LatAngleNorth, LatAngleSouth, LonAngle,
                 MaxDistance, MinDistance, DistanceNormalization, n_particles, device, sim,):
        super(GridSensor3DGrad, self).__init__()
        '''
        SensorName: 传感器名字
        CellScale: 网格尺寸
        GridSize: 网格检测范围（cellArc latAngleSouth latAngleNorth LonAngle maxDistance minDistance DistanceNormalization）
        RotateWithAgent: 是否随Agent旋转
        AgentGameObject: Agent
        AgentID: effetor ID
        DetectableTags: 检测物体body tuple
        MaxColliderBufferSize: 最大检测物数量
        DebugColors: 颜色显示，用于debug
        GizmoZOffset: 沿着Z偏移的尺寸
        ObservationStacks: 时间维度堆叠数量
        DataType: 数据类型 目前支持one-hot

        '''
        # Geometry
        self.SensorName = SensorName
        self.m_AgentGameObject = AgentGameObject
        self.m_ObservationStacks = ObservationStacks
        self.m_CellArc = CellArc
        self.m_LatAngleNorth = LatAngleNorth
        self.m_LatAngleSouth = LatAngleSouth
        self.m_LonAngle = LonAngle
        self.m_MaxDistance = MaxDistance
        self.m_MinDistance = MinDistance
        self.m_DistanceNormalization = DistanceNormalization
        self.dim = 3
        self.n_particles = n_particles
        self.n_bodies = self.m_AgentGameObject.sim.n_bodies
        self.device = device
        # self.agent_groups = self.m_AgentGameObject.sim.agent_groups
        self.dynamics = []
        self.sim = sim
        particle_state = ti.types.struct(
            relative_x=ti.types.vector(self.dim, DTYPE_TI),
            rotated_x=ti.types.vector(self.dim, DTYPE_TI),
            latitudes=DTYPE_TI,
            longitudes=DTYPE_TI,
            distance=DTYPE_TI
        )
        self.s = 0
        grid_sensor = ti.types.struct(
            distance=DTYPE_TI
        )

        self.particle_state = particle_state.field(shape=(33, self.n_particles,), needs_grad=True,
                                                         layout=ti.Layout.SOA)
        self.grid_sensor = grid_sensor.field(shape=(33, (self.m_LonAngle // self.m_CellArc) * 2,
                     (self.m_LatAngleNorth + self.m_LatAngleSouth) // self.m_CellArc, self.n_bodies), needs_grad=True, layout=ti.Layout.SOA)

    @ti.kernel
    def transform_point_particle(self, s: ti.i32, f:ti.i32):
        # 计算point相对agent位置
        for i in range(self.n_particles):
            self.particle_state[s, i].relative_x[0] = self.sim.particles[f, i].x[0] - self.m_AgentGameObject.effectors[0].pos[f][0]
            self.particle_state[s, i].relative_x[1] = self.sim.particles[f, i].x[1] - self.m_AgentGameObject.effectors[0].pos[f][1]
            self.particle_state[s, i].relative_x[2] = self.sim.particles[f, i].x[2] - self.m_AgentGameObject.effectors[0].pos[f][2]

            # 获取四元数数据
            a = self.m_AgentGameObject.effectors[0].quat[f][0]
            b = -self.m_AgentGameObject.effectors[0].quat[f][1]
            c = -self.m_AgentGameObject.effectors[0].quat[f][2]
            d = -self.m_AgentGameObject.effectors[0].quat[f][3]
            rotation_matrix = quaternion_to_rotation_matrix(a, b, c, d)
            self.particle_state[s, i].rotated_x = rotation_matrix @ self.particle_state[s, i].relative_x

    @ti.kernel
    def compute_lat_lon_particle(self, s: ti.i32):
        for i in range(self.n_particles):
            # 提取局部坐标系中的坐标
            x = self.particle_state[s, i].rotated_x[0]
            y = self.particle_state[s, i].rotated_x[1]
            z = self.particle_state[s, i].rotated_x[2]

            # 计算纬度和经度
            # 计算纬度
            self.particle_state[s, i].distance = ti.sqrt(x * x + y * y + z * z)
            cos_lat_rad = ti.max(ti.min(y / self.particle_state[s, i].distance, 1.0), -1.0)
            lat_rad = ti.acos(cos_lat_rad)
            lon_rad = ti.atan2(x, -z)

            self.particle_state[s, i].latitudes = lat_rad * (
                    180.0 / ti.acos(-1.0))  # acos(-1) is a way to get π in Taichi
            self.particle_state[s, i].longitudes = lon_rad * (180.0 / ti.acos(-1.0))

    @ti.kernel
    def normal_distance_particle(self, s: ti.i32):
        # 清空
        # 1. 判断距离是否在球体内
        for p in range(self.n_particles):
            if self.particle_state[s, p].distance < self.m_MaxDistance and self.particle_state[s, p].distance > self.m_MinDistance:
                # 2. 判断经度范围和纬度范围
                if (90 - self.particle_state[s, p].latitudes < self.m_LatAngleNorth and 90 - self.particle_state[s, p].latitudes >= 0) or \
                        (ti.abs(self.particle_state[s, p].latitudes - 90) < self.m_LatAngleSouth and ti.abs(self.particle_state[s, p].latitudes - 90) >= 0):
                    if ti.abs(self.particle_state[s, p].longitudes) < self.m_LonAngle:
                        # 计算加权距离
                        d = (self.particle_state[s, p].distance - self.m_MinDistance) / (self.m_MaxDistance - self.m_MinDistance)
                        normal_d = 0.0
                        if self.m_DistanceNormalization == 1:
                            normal_d = 1 - d
                        else:
                            normal_d = 1 - d / (self.m_DistanceNormalization + ti.abs(d)) * (
                                        self.m_DistanceNormalization + 1)
                        # 计算经纬度索引
                        longitude_index = ti.cast(
                            ti.floor((self.particle_state[s, p].longitudes + self.m_LonAngle) / self.m_CellArc), ti.i32)
                        latitude_index = ti.cast(
                            ti.floor(
                                 (self.particle_state[s, p].latitudes - (90 - self.m_LatAngleNorth)) / self.m_CellArc),
                            ti.i32)

                        # 使用 atomic_max 更新 normal_distance 的值
                        ti.atomic_max(self.grid_sensor[s, longitude_index, latitude_index, self.sim.particles_i[p].body_id].distance, normal_d)
    @ti.kernel
    def get_sensor_data_kernel(self, s: ti.i32, grid_sensor: ti.types.ndarray()):
        # 这里假设 output 已经是一个正确维度和类型的 Taichi field
        for i, j, k in ti.ndrange((self.m_LonAngle // self.m_CellArc) * 2,
                                  (self.m_LatAngleNorth + self.m_LatAngleSouth) // self.m_CellArc,
                                  self.n_bodies):
            grid_sensor[i, j, k] = self.grid_sensor[s, i, j, k].distance

    @ti.kernel
    def clear_grid_sensor(self, s: ti.i32):
        for i, j, k in ti.ndrange((self.m_LonAngle // self.m_CellArc) * 2,
                                  (self.m_LatAngleNorth + self.m_LatAngleSouth) // self.m_CellArc,
                                  self.n_bodies):
            self.grid_sensor[s, i, j, k].distance = 0.0
    def step(self):
        self.s = self.s+1
        self.clear_grid_sensor(self.s)
        self.transform_point_particle(self.s, self.sim.cur_substep_local)
        self.compute_lat_lon_particle(self.s)
        self.normal_distance_particle(self.s)

    def step_grad(self):
        self.normal_distance_particle.grad(self.s)
        self.compute_lat_lon_particle.grad(self.s)
        self.transform_point_particle.grad(self.s, self.sim.cur_substep_local)
        self.clear_grid_sensor.grad(self.s)
        self.s = self.s-1

        # self.debug(self.sim.cur_step_global-1)

    def set_obs(self, s):
        self.transform_point_particle(s, self.sim.cur_substep_local)
        self.compute_lat_lon_particle(s)
        self.normal_distance_particle(s)
    @ti.kernel
    def debug(self, f: ti.i32):
        print(self.sim.particles.grad[f, 0].x[0], self.m_AgentGameObject.effectors[0].pos.grad[f][0], self.particle_state.grad[f, 0].relative_x[0])
        print(self.sim.particles[f, 0].x[0], self.m_AgentGameObject.effectors[0].pos[f][0], self.particle_state[f, 0].relative_x[0])
    def get_obs(self):
        grid_sensor = torch.zeros(((self.m_LonAngle // self.m_CellArc) * 2,
                                   (self.m_LatAngleNorth + self.m_LatAngleSouth) // self.m_CellArc, self.n_bodies), dtype=torch.float32, device=self.device)
        self.get_sensor_data_kernel(self.s, grid_sensor)
        return grid_sensor

    @ti.kernel
    def set_next_state_grad(self, s: ti.i32, grad: ti.types.ndarray()):
        for i, j, k in ti.ndrange((self.m_LonAngle // self.m_CellArc) * 2,
                                  (self.m_LatAngleNorth + self.m_LatAngleSouth) // self.m_CellArc,
                                  self.n_bodies):
            self.grid_sensor.grad[s, i, j, k].distance = grad[i, j, k]

    def reset_grad(self):
        self.particle_state.grad.fill(0)
        self.grid_sensor.grad.fill(0)

    def reset(self):
        self.particle_state.fill(0)
        self.particle_state.grad.fill(0)
        self.grid_sensor.fill(0)
        self.grid_sensor.grad.fill(0)
        self.set_obs(0)