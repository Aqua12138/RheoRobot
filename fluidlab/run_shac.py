import os
import gym
import torch
import random
import argparse
import numpy as np

import fluidlab.envs
from fluidlab.utils.logger import Logger
from fluidlab.optimizer.solver import solve_policy
from fluidlab.optimizer.recorder import record_target, replay_policy, replay_target
from fluidlab.utils.config import load_config
import taichi as ti
# ti.init(arch=ti.gpu, device_memory_GB=8, packed=True, device_memory_fraction=0.9)
ti.init(arch=ti.cpu, packed=True)
import multiprocessing as mp

import yaml
from fluidengine.algorithms.utils.common import *
import fluidengine.algorithms.shac as shac
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg_file", type=str, default=None)
    parser.add_argument("--renderer_type", type=str, default='GGUI')
    parser.add_argument("--logdir", type=str, default="logs/tmp/shac/")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no-time-stamp", action='store_true', default=False)
    parser.add_argument("--train", action='store_true', default=True)



    args = parser.parse_args()

    return args

def main():
    args = get_args()
    with open(args.cfg_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
        # 环境名称

        if not args.no_time_stamp:
            args.logdir = os.path.join(args.logdir, get_time_stamp())
        args.device = torch.device(args.device)
        vargs = vars(args)

        cfg["params"]["general"] = {}
        for key in vargs.keys():
            cfg["params"]["general"][key] = vargs[key]



        traj_optimizer = shac.SHAC(cfg)
        traj_optimizer.train()
        # demo: call other method of env
        # vec_env.env_method("reset_grad")
        # obs = vec_env.reset()


if __name__ == '__main__':
    main()
