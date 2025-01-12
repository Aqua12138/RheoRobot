import numpy as np
from fluidlab.optimizer.optim import *
from fluidlab.utils.misc import is_on_server
import onnx
import torch
from onnx2pytorch import ConvertModel
import torch.optim as optim

if not is_on_server():
    try:
        from pynput import keyboard, mouse
    except:
        pass

class ActionsPolicy:
    def __init__(self, comp_actions):
        self.actions_v = comp_actions[:-1]
        self.actions_p = comp_actions[-1]

    def get_actions_p(self):
        return self.actions_p

    def get_action_v(self, i):
        return self.actions_v[i]

class KeyboardPolicy:
    def __init__(self, init_p=None, v_lin=0.003, v_ang=0.03):
        self.actions_p = init_p
        self.keys_activated = set()
        self.linear_v_mag = v_lin
        self.angular_v_mag = v_ang

        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        try:
            self.keys_activated.add(key.char)
        except:
            pass

    def on_release(self, key):
        try:
            self.keys_activated.remove(key.char)
        except:
            pass

    def get_actions_p(self):
        return self.actions_p


class KeyboardPolicy_vxy_wz(KeyboardPolicy):
    def get_action_v(self, i):
        action_v = np.zeros(6)
        if '4' in self.keys_activated:
            action_v[0] -= self.linear_v_mag
        if '6' in self.keys_activated:
            action_v[0] += self.linear_v_mag
        if '2' in self.keys_activated:
            action_v[1] -= self.linear_v_mag
        if '8' in self.keys_activated:
            action_v[1] += self.linear_v_mag
        if 'x' in self.keys_activated:
            action_v[5] -= self.angular_v_mag
        if 'z' in self.keys_activated:
            action_v[5] += self.angular_v_mag
        return action_v

class KeyboardPolicy_wz(KeyboardPolicy):
    def get_action_v(self, i):
        action_v = np.zeros(6)
        if 'x' in self.keys_activated:
            action_v[5] -= self.angular_v_mag
        if 'z' in self.keys_activated:
            action_v[5] += self.angular_v_mag
        return action_v

class KeyboardPolicy_vxy(KeyboardPolicy):
    def get_action_v(self, i):
        action_v = np.zeros(3)
        if '4' in self.keys_activated:
            action_v[0] -= self.linear_v_mag
        if '6' in self.keys_activated:
            action_v[0] += self.linear_v_mag
        if '2' in self.keys_activated:
            action_v[1] -= self.linear_v_mag
        if '8' in self.keys_activated:
            action_v[1] += self.linear_v_mag
        return action_v


class MousePolicy:
    def __init__(self, init_p):
        self.actions_p      = init_p
        self.mouse_pos      = np.array([0, 0])
        self.mouse_pos_last = None
        self.mouse_pressed  = False
        self.started        = False

        self.listener = mouse.Listener(
            on_move=self.on_move,
            on_click=self.on_click,
        )
        self.listener.start()

    def on_move(self, x, y):
        self.started      = True
        self.mouse_pos[0] = x
        self.mouse_pos[1] = y

    def on_click(self, x, y, button, pressed):
        self.mouse_pos[0]   = x
        self.mouse_pos[1]   = y
        self.mouse_pressed  = pressed

    def get_actions_p(self):
        return self.actions_p

class MousePolicy_vxz(MousePolicy):
    def get_action_v(self, i):
        if not self.started:
            action_v = np.zeros(3)
        else:
            if self.mouse_pos_last is None:
                self.mouse_pos_last = np.array(self.mouse_pos)
            
            mouse_pos_diff = self.mouse_pos - self.mouse_pos_last
            self.mouse_pos_last = np.array(self.mouse_pos)
            action_v = np.array([mouse_pos_diff[0], 0.0, mouse_pos_diff[1]]) * 5e-4

        return action_v


class TrainablePolicy:
    def __init__(self, optim_cfg, init_range, action_dim, horizon, action_range, fix_dim=None):
        self.horizon = horizon
        self.init_range = init_range
        self.action_dim = action_dim
        self.actions_v = np.random.uniform(init_range.v[0], init_range.v[1], size=(horizon, action_dim))
        self.actions_p = np.random.uniform(init_range.p[0], init_range.p[1], size=(action_dim))
        self.action_range = action_range
        self.comp_actions_shape = (horizon+1, action_dim)
        self.trainable = np.full(self.comp_actions_shape[0], True)
        self.fix_dim = fix_dim
        self.freeze_till = 0

        self.optim = eval(optim_cfg.type)(self.comp_actions_shape, optim_cfg)

    @property
    def comp_actions(self):
        return np.vstack([self.actions_v, self.actions_p[None, :]])

    def get_actions_p(self):
        return self.actions_p

    def get_action_v(self, i, **kwargs):
        return self.actions_v[i]

    def optimize(self, grads, loss_info):
        assert grads.shape == self.comp_actions_shape

        grads[np.logical_not(self.trainable)] = 0
        if self.fix_dim is not None:
            grads[:, self.fix_dim] = 0

        new_comp_actions = self.optim.step(self.comp_actions, grads)
        self.actions_p = new_comp_actions[-1]
        self.actions_v = new_comp_actions[:-1].clip(*self.action_range)


class LatteArtPolicy(TrainablePolicy):
    def __init__(self, *args, **kwargs):
        super(LatteArtPolicy, self).__init__(*args, **kwargs)


class LatteArtStirPolicy(TrainablePolicy):
    def __init__(self, *args, **kwargs):
        super(LatteArtStirPolicy, self).__init__(*args, **kwargs)

    def optimize(self, grads, loss_info):
        super(LatteArtStirPolicy, self).optimize(grads, loss_info)

        # task specific processing
        if loss_info['temporal_range'] > 250:
            self.optim.lr = self.optim.init_lr * 0.2
            print(f'lr reduced to {self.optim.lr}')
        elif loss_info['temporal_range'] > 150:
            self.optim.lr = self.optim.init_lr * 0.5
            print(f'lr reduced to {self.optim.lr}')

        for step in [400, 350, 300, 250, 200, 150, 100]:
            if loss_info['temporal_range'] > step:
                freeze_till = step - 100
                self.trainable[:freeze_till] = False
                print(f'feeze till {freeze_till}')
                break


class IceCreamDynamicPolicy(TrainablePolicy):
    def __init__(self, *args, **kwargs):
        super(IceCreamDynamicPolicy, self).__init__(*args, **kwargs)

        self.trainable = np.full(self.comp_actions_shape[0], False)
        self.trainable[169:-1] = True


class IceCreamStaticPolicy(TrainablePolicy):
    def __init__(self, *args, **kwargs):
        super(IceCreamStaticPolicy, self).__init__(*args, **kwargs)
        self.trainable = np.full(self.comp_actions_shape[0], False)
        self.trainable[:-1] = True

    def optimize(self, grads, loss_info):
        grads = grads.clip(-1e5, 1e5)
        super(IceCreamStaticPolicy, self).optimize(grads, loss_info)

        if loss_info['temporal_range'] > 450:
            self.optim.lr = self.optim.init_lr * 0.1
            print(f'lr reduced to {self.optim.lr}')


class GatheringPolicy(TrainablePolicy):
    def __init__(self, *args, **kwargs):
        super(GatheringPolicy, self).__init__(*args, **kwargs)

        self.trainable = np.full(self.comp_actions_shape[0], False)
        self.status = np.full(self.comp_actions_shape[0], 0)
        self.stage_step = [50, 65, 105, 120]
        for i in range(self.horizon):
            if i % self.stage_step[3] < self.stage_step[0]:
                self.trainable[i] = True
                self.status[i] = 0 # moving
            elif self.stage_step[0] <= i % self.stage_step[3] < self.stage_step[1]:
                self.status[i] = 1 # up
            elif self.stage_step[1] <= i % self.stage_step[3] < self.stage_step[2]:
                self.status[i] = 2 # moving back
            elif self.stage_step[2] <= i % self.stage_step[3]:
                self.status[i] = 3 # down

    def get_action_v(self, i, agent=None, update=False):
        if update:
            if self.status[i] == 1:
                self.actions_v[i] = np.array([0, 0.008, 0])
            elif self.status[i] == 2:
                action = (self.actions_p - agent.rigid.latest_pos.to_numpy()[0]) / (self.stage_step[2]- (i % self.stage_step[3]))
                action[1] = 0
                self.actions_v[i] = action
            elif self.status[i] == 3:
                self.actions_v[i] = np.array([0, -0.008, 0])
            # elif self.status[i] == 0:
            #     self.actions_v[i] = np.array([0.003, 0, 0])

        return self.actions_v[i]

    def optimize(self, grads, loss_info):
        for step in [720, 600, 480, 360, 240, 120]:
            if loss_info['temporal_range'] > step:
                self.freeze_till = loss_info['temporal_range'] - 120
                self.trainable[:self.freeze_till] = False
                print(f'feeze till {self.freeze_till}')
                break

        super(GatheringPolicy, self).optimize(grads, loss_info)


class GatheringOPolicy(TrainablePolicy):
    def __init__(self, *args, **kwargs):
        super(GatheringOPolicy, self).__init__(*args, **kwargs)

        self.trainable = np.full(self.comp_actions_shape[0], False)
        self.status = np.full(self.comp_actions_shape[0], 0)
        self.stage_step = [50, 65, 105, 120]
        for i in range(self.horizon):
            if i % self.stage_step[3] < self.stage_step[0]:
                self.trainable[i] = True
                self.status[i] = 0 # moving
            elif self.stage_step[0] <= i % self.stage_step[3] < self.stage_step[1]:
                self.status[i] = 1 # up
            elif self.stage_step[1] <= i % self.stage_step[3] < self.stage_step[2]:
                self.status[i] = 2 # moving back
            elif self.stage_step[2] <= i % self.stage_step[3]:
                self.status[i] = 3 # down

    def get_action_v(self, i, agent=None, update=False):
        if update:
            if self.status[i] == 1:
                self.actions_v[i] = np.array([0, 0.008, 0])
            elif self.status[i] == 2:
                action = (self.actions_p - agent.rigid.latest_pos.to_numpy()[0]) / (self.stage_step[2]- (i % self.stage_step[3]))
                action[1] = 0
                self.actions_v[i] = action
            elif self.status[i] == 3:
                self.actions_v[i] = np.array([0, -0.008, 0])
            # elif self.status[i] == 0:
            #     self.actions_v[i] = np.array([-0.004, 0, 0])

        return self.actions_v[i]


    def optimize(self, grads, loss_info):
        super(GatheringOPolicy, self).optimize(grads, loss_info)

        # for step in [720, 600, 480, 360, 240, 120]:
        #     if loss_info['temporal_range'] > step:
        #         self.freeze_till = loss_info['temporal_range'] - 120
        #         self.trainable[:self.freeze_till] = False
        #         print(f'feeze till {self.freeze_till}')
        #         break

class MixingPolicy(TrainablePolicy):
    def __init__(self, *args, **kwargs):
        super(MixingPolicy, self).__init__(*args, **kwargs)

        self.trainable = np.full(self.comp_actions_shape[0], False)
        self.status = np.full(self.comp_actions_shape[0], 0)
        self.stage_step = [50, 80]
        for i in range(self.horizon):
            if i % self.stage_step[1] < self.stage_step[0]:
                self.trainable[i] = True
                self.status[i] = 0 # moving
            elif self.stage_step[0] <= i % self.stage_step[1]:
                self.status[i] = 1 # moving back

    def get_action_v(self, i, agent=None, update=False):
        if update:
            if self.status[i] == 1:
                action = (np.array([0.5, 0.73, 0.5]) - agent.rigid.latest_pos.to_numpy()[0]) / (self.stage_step[1]- (i % self.stage_step[1]))
                self.actions_v[i] = action
            # elif self.status[i] == 0:
            #     self.actions_v[i] = np.array([-0.005, 0, 0])

        return self.actions_v[i]

    def optimize(self, grads, loss_info):
        super(MixingPolicy, self).optimize(grads, loss_info)

        steps = list(range(80, 2000, 80))[::-1]
        for step in steps:
            if loss_info['temporal_range'] > step:
                self.freeze_till = loss_info['temporal_range'] - 160
                self.trainable[:self.freeze_till] = False
                print(f'feeze till {self.freeze_till}')
                break

class CirculationPolicy(TrainablePolicy):
    def __init__(self, *args, **kwargs):
        super(CirculationPolicy, self).__init__(*args, **kwargs)


    def optimize(self, grads, loss_info):
        super(CirculationPolicy, self).optimize(grads, loss_info)

        # if loss_info['iteration'] > 150:
        #     self.optim.lr = self.optim.init_lr * 0.1
        #     print(f'lr reduced to {self.optim.lr}')

        # elif loss_info['iteration'] > 100:
        #     self.optim.lr = self.optim.init_lr * 0.25
        #     print(f'lr reduced to {self.optim.lr}')

class PouringPolicy(TrainablePolicy):
    def __init__(self, *args, **kwargs):
        super(PouringPolicy, self).__init__(*args, **kwargs)


class TransportingPolicy(TrainablePolicy):
    def __init__(self, *args, **kwargs):
        super(TransportingPolicy, self).__init__(*args, **kwargs)
        self.trainable = np.full(self.comp_actions_shape[0], False)
        self.trainable[:-1] = True

class TorchGatheringPolicy(TrainablePolicy):
    def __init__(self, *args, **kwargs):
        super(TorchGatheringPolicy, self).__init__(*args, **kwargs)
        # # fluidlab policy
        # self.trainable = np.full(self.comp_actions_shape[0], False)
        # self.status = np.full(self.comp_actions_shape[0], 0)
        # self.stage_step = [50, 65, 105, 120] # 120 为一个动作周期
        # for i in range(self.horizon):
        #     if i % self.stage_step[3] < self.stage_step[0]:
        #         self.trainable[i] = True
        #         self.status[i] = 0 # moving
        #     elif self.stage_step[0] <= i % self.stage_step[3] < self.stage_step[1]:
        #         self.status[i] = 1 # up
        #     elif self.stage_step[1] <= i % self.stage_step[3] < self.stage_step[2]:
        #         self.status[i] = 2 # moving back
        #     elif self.stage_step[2] <= i % self.stage_step[3]:
        #         self.status[i] = 3 # down

        # my policy
        onnx_model_path = "/home/zhx/Project/results/2024_4_7_1647/Collection/Collection-4999936.onnx"
        onnx_model = onnx.load(onnx_model_path)
        self.pytorch_model = ConvertModel(onnx_model)

        # self.pytorch_model = torch.load('/home/zhx/PycharmProjects/fluids/FluidLab_3_28/fluidlab/optimizer/model/gatheringEasy/model.pth')
        # self.pytorch_model.train()
        print(self.pytorch_model)

        self.optimizer = optim.Adam(self.pytorch_model.parameters(), lr=1e-5)

        # for torch
        self.actions_v_torch = 4000 * [None]
        self.actions_p_torch = [None]

        self.clip_norm = 1

    def get_action_v(self, i, agent=None, update=False):
        # fluidlab
        # if update:
        #     # policy
        #     if self.status[i] == 1:
        #         self.actions_v[i] = np.array([0, 0.008, 0])
        #     elif self.status[i] == 2:
        #         action = (self.actions_p - agent.rigid.latest_pos.to_numpy()[0]) / (self.stage_step[2]- (i % self.stage_step[3]))
        #         action[1] = 0
        #         self.actions_v[i] = action
        #     elif self.status[i] == 3:
        #         self.actions_v[i] = np.array([0, -0.008, 0])
        #     # elif self.status[i] == 0:
        #     #     self.actions_v[i] = np.array([0.003, 0, 0])

        # My
        if update:
            # get agent state
            # obs = np.array(agent.get_state(i))
            # obs = torch.from_numpy(obs).float()
            # get visual state
            obs = agent.get_obs()
            obs_visual = torch.from_numpy(obs[0].transpose(0, 3, 1, 2)).float()
            obs_vector = torch.from_numpy(obs[1]).float()
            # 与actions_v同步维护
            a = self.pytorch_model(obs_visual, obs_vector)
            self.actions_v_torch[i] = self.pytorch_model(obs_visual, obs_vector)[4][0]
            # filename = "/home/zhx/Project/FluidRobotic/fluidlab/tmp/obs/play/frame_{:03d}.npy".format(i)
            # np.save(filename, obs_visual)
            # print("save", i)

            # 维护执行动作
            action = self.actions_v_torch[i].clone()
            action_numpy = action.detach().numpy()
            action_numpy = action_numpy * 0.35 * 2e-2
            action_numpy[0] *= -1
            action_numpy[1] *= 1
            action_numpy[2] *= 1
            action_numpy[3] *= -3
            # action_numpy[4] *= -1
            # action_numpy[5] *= 1
            self.actions_v[i] = np.array([action_numpy[1], action_numpy[2], action_numpy[0],
                                          0, action_numpy[3], 0])

        return self.actions_v[i]

    def optimize(self, grads, loss_info):
        # for step in [720, 600, 480, 360, 240, 120]:
        #     if loss_info['temporal_range'] > step:
        #         self.freeze_till = loss_info['temporal_range'] - 120
        #         self.trainable[:self.freeze_till] = False
        #         print(f'feeze till {self.freeze_till}')
        #         break

        # # fluidlab
        # assert grads.shape == self.comp_actions_shape
        # grads[np.logical_not(self.trainable)] = 0
        # if self.fix_dim is not None:
        #     grads[:, self.fix_dim] = 0
        #
        # new_comp_actions = self.optim.step(self.comp_actions, grads)
        # self.actions_p = new_comp_actions[-1]
        # self.actions_v = new_comp_actions[:-1].clip(*self.action_range)

        # # My
        assert grads.shape == self.comp_actions_shape
        self.optimizer.zero_grad()  # 清空梯度，准备下一轮优化
        for i in range(self.horizon):
            adjusted_grad = torch.tensor([-grads[i][2],  # Reorder and adjust signs as per actions
                                           grads[i][0],
                                           grads[i][1],# Reorder and adjust signs as per actions
                                           -grads[i][3],
                                           ], dtype=torch.float32)
            grad_norm = np.linalg.norm(adjusted_grad)
            if grad_norm > self.clip_norm:
                adjusted_grad = adjusted_grad * (self.clip_norm / grad_norm)
            self.actions_v_torch[i].backward(adjusted_grad)

        # self.actions[i].backward(grads)
        self.optimizer.step()  # 执行参数更新

    def get_actions_p(self):
        self.actions_p = np.random.uniform(self.init_range.p[0], self.init_range.p[1], size=(self.action_dim))
        return self.actions_p
