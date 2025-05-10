# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from ast import arg
from matplotlib.pyplot import get
import numpy as np
import random

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task
from utils.process_sarl import *
from utils.process_offrl import *

def train():
    print("Algorithm: ", args.algo)


    if args.algo in ["ppo","ppo_pnorm","ppo_max","ppo_stability", "ppo_add_intrisic", "ppo_add_stability", "ppo_add_stability_constrain", "ppo_add_feature", "ppo_statenorm", "ppo_lyp", "ppo_lyp_constrain", "ddpg","sac","td3","trpo"]:
        task, env = parse_task(args, cfg, cfg_train, sim_params)

        sarl = eval('process_{}'.format(args.algo))(args, env, cfg_train, logdir)

        iterations = cfg_train["learn"]["max_iterations"]
        if args.max_iterations > 0:
            iterations = args.max_iterations

        sarl.run(num_learning_iterations=iterations, log_interval=cfg_train["learn"]["save_interval"])
    
    elif args.algo in ["td3_bc", "bcq", "iql", "ppo_collect",'gail']:
        task, env = parse_task(args, cfg, cfg_train, sim_params)

        offrl = eval('process_{}'.format(args.algo))(args, env, cfg_train, logdir)

        iterations = cfg_train["learn"]["max_iterations"]
        if args.max_iterations > 0:
            iterations = args.max_iterations

        offrl.run(num_learning_iterations=iterations, log_interval=cfg_train["learn"]["save_interval"])
    
    else:
        print("Unrecognized algorithm!\nAlgorithm should be one of: [sac,td3,trpo,ppo,ddpg,td3_bc, bcq, iql, ppo_collect]")


if __name__ == '__main__':
    set_np_formatting()
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    print('args\n', args, '\n')
    print('cfg\n', cfg, '\n')
    print('cfg_train\n', cfg_train, '\n')
    print('sim_params\n', sim_params, '\n')
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    train()
