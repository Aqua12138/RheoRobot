import numpy as np
import taichi as ti
import torch
from fluidlab.configs.macros import *
from pyquaternion import Quaternion
from .gridSensor import GridSensor

@ti.data_oriented
class GridSensor2D(GridSensor):
    def __init__(self, SensorName, CellScale, GridSize, RotateWithAgent, AgentGameObject, ObservationStacks, n_particles, GlobalPosition=None):
        super(GridSensor2D, self).__init__()
        '''
        SensorName: 传感器名字
        CellScale: 网格尺寸
        GridSize: 网格检测范围
        RotateWithAgent: 是否随Agent旋转
        AgentGameObject: Agent
        DetectableTags: 检测物体body tuple
        MaxColliderBufferSize: 最大检测物数量
        DebugColors: 颜色显示，用于debug
        GizmoZOffset: 沿着Z偏移的尺寸
        ObservationStacks: 时间维度堆叠数量
        DataType: 数据类型 目前支持one-hot
        '''
        self.SensorName = SensorName
        self.m_CellScale = CellScale
        self.m_GridSize = GridSize

        self.m_AgentGameObject = AgentGameObject
        self.m_ObservationStacks = ObservationStacks
        self.dim = 3
        self.n_particles = n_particles
        self.statics = self.m_AgentGameObject.sim.statics
        self.n_statics = self.m_AgentGameObject.sim.n_statics
        self.n_bodies = self.m_AgentGameObject.sim.n_bodies
        self.inv_CellScale = tuple(1/x for x in self.m_CellScale)
        self.m_GlobalPosition = GlobalPosition
        if self.m_GlobalPosition is not None:
            self.m_RotateWithAgent = False
        else:
            self.m_RotateWithAgent = RotateWithAgent
        self.resolution = (64, 64, 64)  # for check mesh
        self.cell_size = 1 / 64

        self.f = 0
        # self.agent_groups = self.m_AgentGameObject.sim.agent_groups
        self.dynamics = []
        # self.setup_dynamic_mesh()
        self.n_dynamics = len(self.dynamics)

        particle_state_RL = ti.types.struct(
            x = ti.types.vector(self.dim, DTYPE_TI),
            used=ti.i32,
            tag=ti.i32,
            relative_x = ti.types.vector(self.dim, DTYPE_TI),
            rotated_x = ti.types.vector(self.dim, DTYPE_TI),
            grid_x = ti.types.vector(2, ti.i32),
        )

        mesh_state_RL = ti.types.struct(
            x=ti.types.vector(self.dim, DTYPE_TI),
            used=ti.i32,
            tag=ti.i32,
            relative_x=ti.types.vector(self.dim, DTYPE_TI),
            rotated_x=ti.types.vector(self.dim, DTYPE_TI),
            grid_x=ti.types.vector(2, ti.i32),
        )

        agent_state_RL = ti.types.struct(
            agent_position = ti.types.vector(self.dim, DTYPE_TI),
            agent_rotation = ti.types.matrix(self.dim, self.dim, DTYPE_TI)
        )

        self.particle_state_RL = particle_state_RL.field(shape=(self.n_particles,), needs_grad=False,
                                                         layout=ti.Layout.SOA)  # 用于更新 Grid sensor
        self.mesh_state = mesh_state_RL.field(shape=self.resolution, needs_grad=False,
                                              layout=ti.Layout.SOA)  # 用于更新 Grid sensor
        self.agent_state_RL = agent_state_RL.field(shape=(), needs_grad=False,
                                                         layout=ti.Layout.SOA)  # 用于更新 Grid sensor
        # init kernel
        self.init_mesh_kernel()

    @ti.kernel
    def init_mesh_kernel(self):
        for I in ti.grouped(ti.ndrange(*self.resolution)):
            p = I * self.cell_size
            for i in ti.static(range(self.dim)):
                self.mesh_state[I].x[i] = p[i]

    def setup_dynamic_mesh(self):
        # 把所有非自身的agent.rigid对象提取出来
        for group in self.agent_groups:
            for agent in group.agents:
                # if is local, the self can not be check
                if self.m_GlobalPosition is None:
                    if agent != self.m_AgentGameObject:
                        self.dynamics.append(agent.rigid.mesh)
                else:
                    self.dynamics.append(agent.rigid.mesh)

    @ti.kernel
    def transform_point(self, x: ti.f32, y: ti.f32, z: ti.f32, agent_rotation: ti.types.matrix(3, 3, ti.f32)):
        # particle
        self.transform_point_particle(x, y, z, agent_rotation)
        # mesh
        self.transform_point_mesh(x, y, z, agent_rotation)

    @ti.func
    def transform_point_particle(self, x, y, z, agent_rotation):
        # 计算point相对agent位置
        for i in range(self.n_particles):
            self.particle_state_RL[i].relative_x[0] = self.particle_state_RL[i].x[0] - x
            self.particle_state_RL[i].relative_x[1] = self.particle_state_RL[i].x[1] - y
            self.particle_state_RL[i].relative_x[2] = self.particle_state_RL[i].x[2] - z
            self.particle_state_RL[i].rotated_x = agent_rotation @ self.particle_state_RL[i].relative_x

    @ti.func
    def transform_point_mesh(self, x, y, z, agent_rotation):
        for I in ti.grouped(ti.ndrange(*self.resolution)):
            self.mesh_state[I].relative_x[0] = self.mesh_state[I].x[0] - x
            self.mesh_state[I].relative_x[1] = self.mesh_state[I].x[1] - y
            self.mesh_state[I].relative_x[2] = self.mesh_state[I].x[2] - z
            self.mesh_state[I].rotated_x = agent_rotation @ self.mesh_state[I].relative_x

    @ti.kernel
    def one_hot_gird_particle(self, one_hot: ti.types.ndarray()):
        # 2 判断 a) 偏移位置是否在0～m_GridSize-1 内 b)
        # 按照tag 写入对应通道
        for p in range(self.n_particles):
            if (0 <= self.particle_state_RL[p].grid_x[0] < self.m_GridSize[0]) and \
                    (0 <= self.particle_state_RL[p].grid_x[1] < self.m_GridSize[1]):
                one_hot[ti.cast(self.particle_state_RL[p].grid_x[0], ti.i32), ti.cast(self.particle_state_RL[p].grid_x[1], ti.i32), self.particle_state_RL[p].tag] = 1

    @ti.kernel
    def one_hot_gird_mesh(self, one_hot: ti.types.ndarray()):
        for I in ti.grouped(ti.ndrange(*self.resolution)):
            if (0 <= self.mesh_state[I].grid_x[0] < self.m_GridSize[0]) and \
                    (0 <= self.mesh_state[I].grid_x[1] < self.m_GridSize[1]):
                for i in ti.static(range(self.n_statics)):
                    if self.statics[i].is_collide(self.mesh_state[I].x):
                        one_hot[ti.cast(self.mesh_state[I].grid_x[0], ti.i32), ti.cast(self.mesh_state[I].grid_x[1], ti.i32), i] = 1
                for i in ti.static(range(self.n_dynamics)):
                    if self.dynamics[i].is_collide(self.f, self.mesh_state[I].x):
                        one_hot[ti.cast(self.mesh_state[I].grid_x[0], ti.i32), ti.cast(self.mesh_state[I].grid_x[1], ti.i32), i+self.n_statics] = 1

    @ti.kernel
    def density_gird_particle(self, denisty: ti.types.ndarray()):
        # 2 判断 a) 偏移位置是否在0～m_GridSize-1 内 b)
        # 按照tag 写入对应通道
        for p in range(self.n_particles):
            if (0 <= self.particle_state_RL[p].grid_x[0] < self.m_GridSize[0]) and \
                    (0 <= self.particle_state_RL[p].grid_x[1] < self.m_GridSize[1]):
                # a = self.particle_state_RL[p].grid_x[1]
                denisty[ti.cast(self.particle_state_RL[p].grid_x[0], ti.i32), ti.cast(self.particle_state_RL[p].grid_x[1], ti.i32), self.particle_state_RL[p].tag] += 1

    @ti.kernel
    def density_gird_mesh(self, denisty: ti.types.ndarray()):
        for I in ti.grouped(ti.ndrange(*self.resolution)):
            if (0 <= self.mesh_state[I].grid_x[0] < self.m_GridSize[0]) and \
                    (0 <= self.mesh_state[I].grid_x[1] < self.m_GridSize[1]):
                # a = self.particle_state_RL[p].grid_x[1]
                for i in ti.static(range(self.n_statics)):
                    if self.statics[i].is_collide(self.mesh_state[I].x) == 1:
                        denisty[ti.cast(self.mesh_state[I].grid_x[0], ti.i32), ti.cast(self.mesh_state[I].grid_x[1], ti.i32), i] += 1
                for i in ti.static(range(self.n_dynamics)):
                    if self.dynamics[i].is_collide(self.f, self.mesh_state[I].x):
                        denisty[ti.cast(self.mesh_state[I].grid_x[0], ti.i32), ti.cast(self.mesh_state[I].grid_x[1], ti.i32), i+self.n_statics] += 1
    @ti.kernel
    def transform_grid(self):
        self.transform_grid_particle()
        self.transform_grid_mesh()

    @ti.func
    def transform_grid_particle(self):
        # 将相对位置转移到网格坐标
        if self.m_RotateWithAgent:
            for p in range(self.n_particles):
                for i in ti.static(range(2)):
                    self.particle_state_RL[p].grid_x[i] = int(ti.floor(self.particle_state_RL[p].rotated_x[2 * i] * self.inv_CellScale[i] - 0.5))
                    # 更新grid状态到网格中
                    # 1 位置偏移
                    self.particle_state_RL[p].grid_x[i] += int(ti.floor(self.m_GridSize[i] / 2))
        else:
            for p in range(self.n_particles):
                if self.particle_state_RL[p].used:
                    for i in ti.static(range(2)):
                        self.particle_state_RL[p].grid_x[i] = int(ti.floor(self.particle_state_RL[p].relative_x[2 * i] * self.inv_CellScale[i] - 0.5))
                        # 更新grid状态到网格中
                        # 1 位置偏移
                        self.particle_state_RL[p].grid_x[i] += int(ti.floor(self.m_GridSize[i] / 2))
    @ti.func
    def transform_grid_mesh(self):
        # 将相对位置转移到网格坐标
        if self.m_RotateWithAgent:
            for I in ti.grouped(ti.ndrange(*self.resolution)):
                for i in ti.static(range(2)):
                    self.mesh_state[I].grid_x[i] = int(
                        ti.floor(self.mesh_state[I].rotated_x[2 * i] * self.inv_CellScale[i] - 0.5))
                    # 更新grid状态到网格中
                    # 1 位置偏移
                    self.mesh_state[I].grid_x[i] += int(ti.floor(self.m_GridSize[i] / 2))
        else:
            for I in ti.grouped(ti.ndrange(*self.resolution)):
                for i in ti.static(range(2)):
                    self.mesh_state[I].grid_x[i] = int(
                        ti.floor(self.mesh_state[I].rotated_x[2 * i] * self.inv_CellScale[i] - 0.5))
                    # 更新grid状态到网格中
                    # 1 位置偏移
                    self.mesh_state[I].grid_x[i] += int(ti.floor(self.m_GridSize[i] / 2))

    def UpdateSensor(self, RL_state, f):
        agent_state = self.m_AgentGameObject.get_state(f)
        # 1、更新颗粒和agent数据
        self.particle_state_RL.x = RL_state.x
        self.particle_state_RL.used = RL_state.used
        self.particle_state_RL.tag = RL_state.body_id

        # 2 计算agent坐标矩阵变换
        if self.m_GlobalPosition is None:
            q = Quaternion(agent_state[0][3], -agent_state[0][4], agent_state[0][5], agent_state[0][6])
            rotation_matrix = q.rotation_matrix # R
            # 转换为矩阵
            agent_rotation = ti.Matrix(rotation_matrix, dt=DTYPE_TI)
            # 3 计算颗粒的相对位置
            self.transform_point(float(agent_state[0][0]), float(agent_state[0][1]), float(agent_state[0][2]), agent_rotation)
        else:
            agent_rotation = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dt=DTYPE_TI)
            self.transform_point(self.m_GlobalPosition[0], self.m_GlobalPosition[1], self.m_GlobalPosition[2], agent_rotation)
        # 4 相对位置转换成网格坐标
        self.transform_grid()

    def get_obs(self, obs_type = "one-hot"):
        particle_state = np.zeros((*self.m_GridSize, self.n_bodies), dtype=DTYPE_NP)
        mesh_state = np.zeros((*self.m_GridSize, self.n_statics+self.n_dynamics), dtype=DTYPE_NP)
        if obs_type == "one-hot":
            self.one_hot_gird_particle(particle_state)
            self.one_hot_gird_mesh(mesh_state)
        elif obs_type == "densityMap":
            self.density_gird_particle(particle_state)
            self.density_gird_mesh(mesh_state)
        state = np.concatenate((mesh_state, particle_state), axis=-1, dtype=DTYPE_NP)
        return state



