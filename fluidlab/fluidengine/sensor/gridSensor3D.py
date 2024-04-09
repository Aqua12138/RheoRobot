import numpy as np
import taichi as ti
import torch
from fluidlab.configs.macros import *
from pyquaternion import Quaternion
from .gridSensor import GridSensor


@ti.data_oriented
class GridSensor3D(GridSensor):
    def __init__(self, SensorName, AgentGameObject, ObservationStacks, CellArc, LatAngleNorth, LatAngleSouth, LonAngle,
                 MaxDistance, MinDistance, DistanceNormalization, n_particles):
        super(GridSensor3D, self).__init__()
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
        self.statics = self.m_AgentGameObject.sim.statics
        self.n_statics = self.m_AgentGameObject.sim.n_statics
        self.n_bodies = self.m_AgentGameObject.sim.n_bodies
        # self.agent_groups = self.m_AgentGameObject.sim.agent_groups
        self.dynamics = []
        # self.setup_dynamic_mesh()
        # self.n_dynamics = len(self.dynamics)
        self.f = 0
        particle_state_RL = ti.types.struct(
            x=ti.types.vector(self.dim, DTYPE_TI),
            used=ti.i32,
            tag=ti.i32,
            relative_x=ti.types.vector(self.dim, DTYPE_TI),
            rotated_x=ti.types.vector(self.dim, DTYPE_TI),
            latitudes=DTYPE_TI,
            longitudes=DTYPE_TI,
            distance=DTYPE_TI
        )
        agent_state_RL = ti.types.struct(
            agent_position=ti.types.vector(self.dim, DTYPE_TI),
            agent_rotation=ti.types.matrix(self.dim, self.dim, DTYPE_TI)
        )
        mesh_state_RL = ti.types.struct(
            x=ti.types.vector(self.dim, DTYPE_TI),
            used=ti.i32,
            tag=ti.i32,
            relative_x=ti.types.vector(self.dim, DTYPE_TI),
            rotated_x=ti.types.vector(self.dim, DTYPE_TI),
            latitudes=DTYPE_TI,
            longitudes=DTYPE_TI,
            distance=DTYPE_TI
        )
        self.particle_state_RL = particle_state_RL.field(shape=(self.n_particles,), needs_grad=False,
                                                         layout=ti.Layout.SOA)  # 用于更新 Grid sensor
        self.agent_state_RL = agent_state_RL.field(shape=(), needs_grad=False,
                                                   layout=ti.Layout.SOA)  # 用于更新 Grid sensor
        # Assume max_statics is the maximum number of statics you expect to handle
        self.n_vertexs = 0
        for i in range(len(self.statics)):
            self.n_vertexs += self.statics[i].n_vertices
        self.resolution = (30, 30, 30)  # for check mesh
        self.cell_size = 1 / 30
        self.mesh_state = mesh_state_RL.field(shape=self.resolution, needs_grad=False,
                                              layout=ti.Layout.SOA)  # 用于更新 Grid sensor
        # 从 Taichi MatrixField 转换到 numpy 数组
        self.current_vertex = ti.field(dtype=ti.i32, shape=())
        self.current_vertex[None] = 0
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
                if agent != self.m_AgentGameObject:
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
    def compute_lat_lon(self):
        # particle
        self.compute_lat_lon_particle()
        # mesh
        self.compute_lat_lon_mesh()

    @ti.func
    def compute_lat_lon_particle(self):
        for i in range(self.n_particles):
            # 提取局部坐标系中的坐标
            x = self.particle_state_RL[i].rotated_x[0]
            y = self.particle_state_RL[i].rotated_x[1]
            z = self.particle_state_RL[i].rotated_x[2]

            # 计算纬度和经度
            # 计算纬度
            self.particle_state_RL[i].distance = ti.sqrt(x * x + y * y + z * z)
            cos_lat_rad = y / self.particle_state_RL[i].distance
            lat_rad = ti.acos(cos_lat_rad)
            lon_rad = ti.atan2(x, -z)
            # Convert radians to degrees

            self.particle_state_RL[i].latitudes = lat_rad * (
                    180.0 / ti.acos(-1.0))  # acos(-1) is a way to get π in Taichi
            self.particle_state_RL[i].longitudes = lon_rad * (180.0 / ti.acos(-1.0))

    @ti.func
    def compute_lat_lon_mesh(self):
        for I in ti.grouped(ti.ndrange(*self.resolution)):
            x = self.mesh_state[I].rotated_x[0]
            y = self.mesh_state[I].rotated_x[1]
            z = self.mesh_state[I].rotated_x[2]

            # 计算纬度和经度
            # 计算纬度
            self.mesh_state[I].distance = ti.sqrt(x * x + y * y + z * z)
            cos_lat_rad = y / self.mesh_state[I].distance
            lat_rad = ti.acos(cos_lat_rad)
            lon_rad = ti.atan2(x, -z)
            # Convert radians to degrees

            self.mesh_state[I].latitudes = lat_rad * (
                    180.0 / ti.acos(-1.0))  # acos(-1) is a way to get π in Taichi
            self.mesh_state[I].longitudes = lon_rad * (180.0 / ti.acos(-1.0))

    @ti.kernel
    def normal_distance_particle(self, particle_state: ti.types.ndarray()):
        # 1. 判断距离是否在球体内
        for p in range(self.n_particles):
            if self.particle_state_RL[p].distance < self.m_MaxDistance and self.particle_state_RL[
                p].distance > self.m_MinDistance:

                # 2. 判断经度范围和纬度范围
                if (90 - self.particle_state_RL[p].latitudes < self.m_LatAngleNorth and 90 - self.particle_state_RL[
                    p].latitudes >= 0) or \
                        (ti.abs(self.particle_state_RL[p].latitudes - 90) < self.m_LatAngleSouth and ti.abs(
                            self.particle_state_RL[p].latitudes - 90) >= 0):
                    if ti.abs(self.particle_state_RL[p].longitudes) < self.m_LonAngle:
                        # 计算加权距离
                        d = (self.particle_state_RL[p].distance - self.m_MinDistance) / (
                                self.m_MaxDistance - self.m_MinDistance)
                        normal_d = 0.0
                        if self.m_DistanceNormalization == 1:
                            normal_d = 1 - d
                        else:
                            normal_d = 1 - d / (self.m_DistanceNormalization + ti.abs(d)) * (
                                        self.m_DistanceNormalization + 1)
                        # 计算经纬度索引
                        longitude_index = ti.cast(
                            ti.floor((self.particle_state_RL[p].longitudes + self.m_LonAngle) / self.m_CellArc), ti.i32)
                        latitude_index = ti.cast(
                            ti.floor(
                                (self.particle_state_RL[p].latitudes - (90 - self.m_LatAngleNorth)) / self.m_CellArc),
                            ti.i32)

                        # 使用 atomic_max 更新 normal_distance 的值
                        ti.atomic_max(particle_state[longitude_index, latitude_index, self.particle_state_RL[p].tag],
                                      normal_d)

    @ti.kernel
    def normal_distance_mesh(self, mesh_state: ti.types.ndarray()):
        for I in ti.grouped(ti.ndrange(*self.resolution)):
            if self.mesh_state[I].distance < self.m_MaxDistance and self.mesh_state[I].distance > self.m_MinDistance:
                # 2. 判断经度范围和纬度范围
                if (90 - self.mesh_state[I].latitudes < self.m_LatAngleNorth and 90 - self.mesh_state[
                    I].latitudes >= 0) or \
                        (ti.abs(self.mesh_state[I].latitudes - 90) < self.m_LatAngleSouth and ti.abs(
                            self.mesh_state[I].latitudes - 90) >= 0):
                    if ti.abs(self.mesh_state[I].longitudes) < self.m_LonAngle:
                        # 计算加权距离
                        d = (self.mesh_state[I].distance - self.m_MinDistance) / (
                                self.m_MaxDistance - self.m_MinDistance)
                        normal_d = 0.0
                        if self.m_DistanceNormalization == 1:
                            normal_d = 1 - d
                        else:
                            normal_d = 1 - d / (self.m_DistanceNormalization + ti.abs(d)) * (
                                    self.m_DistanceNormalization + 1)

                        # 计算经纬度索引
                        longitude_index = ti.cast(
                            ti.floor((self.mesh_state[I].longitudes + self.m_LonAngle) / self.m_CellArc),
                            ti.i32)
                        latitude_index = ti.cast(
                            ti.floor(
                                (self.mesh_state[I].latitudes - (
                                        90 - self.m_LatAngleNorth)) / self.m_CellArc),
                            ti.i32)

                        for i in ti.static(range(self.n_statics)):
                            ti.atomic_max(mesh_state[longitude_index, latitude_index, i],
                                          normal_d * self.statics[i].is_collide(self.mesh_state[I].x))

                        # for i in ti.static(range(self.n_dynamics)):
                        #     ti.atomic_max(mesh_state[longitude_index, latitude_index, i + self.n_statics],
                        #                   normal_d * self.dynamics[i].is_collide(self.f, self.mesh_state[I].x))

    def UpdateSensor(self, RL_state, f):
        self.f = f
        agent_state = self.m_AgentGameObject.get_state(f)
        # 1、更新颗粒和agent数据2
        self.particle_state_RL.x = RL_state.x
        self.particle_state_RL.used = RL_state.used
        self.particle_state_RL.tag = RL_state.body_id
        # print("Local coordinate system X:", self.particle_state_RL[0].rotated_x)
        # 2.1 计算agent坐标矩阵变换
        q = Quaternion(agent_state[0][3], -agent_state[0][4], -agent_state[0][5],
                       -agent_state[0][6])
        rotation_matrix = q.rotation_matrix  # R
        # 2.2 转换为矩阵
        agent_rotation = ti.Matrix(rotation_matrix, dt=DTYPE_TI)
        # 3 计算颗粒的相对位置
        self.transform_point(float(agent_state[0][0]), float(agent_state[0][1]),
                             float(agent_state[0][2]), agent_rotation)
        # 4 计算经纬
        self.compute_lat_lon()

    def get_obs(self, obs_type="NormalDistance"):
        particle_state = np.zeros(((self.m_LonAngle // self.m_CellArc) * 2,
                                   (self.m_LatAngleNorth + self.m_LatAngleSouth) // self.m_CellArc, self.n_bodies),
                                  dtype=DTYPE_NP)
        # mesh_state = np.zeros(((self.m_LonAngle // self.m_CellArc) * 2,
        #                        (self.m_LatAngleNorth + self.m_LatAngleSouth) // self.m_CellArc,
        #                        self.n_statics + self.n_dynamics), dtype=DTYPE_NP)
        if obs_type == "NormalDistance":
            self.normal_distance_particle(particle_state)
        #     self.normal_distance_mesh(mesh_state)
        # state = np.concatenate((mesh_state, particle_state), axis=-1)
        return np.flip(particle_state.transpose(1, 0, 2), 0)

