# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from tasks.shadow_hand import ShadowHand
from tasks.humanoid import Humanoid
from tasks.shadow_hand_test import ShadowHandTest
from tasks.ant import Ant
from tasks.half_cheetah import HalfCheetah
from tasks.hopper import Hopper
from tasks.swimmer import Swimmer
from tasks.walker2d import Walker
from tasks.ball_balance import BallBalance
from tasks.cartpole import Cartpole
from tasks.franka_cabinet import FrankaCabinet
from tasks.franka_cube_stack import FrankaCubeStack
from tasks.shadow_hand_catch_overarm import ShadowHandCatchOverarm
# from tasks.shadow_hand_catch_overarm_allobj import ShadowHandCatchOverarm
from tasks.shadow_hand_catch_underarm import ShadowHandCatchUnderarm
from tasks.shadow_hand_catch_abreast import ShadowHandCatchAbreast
from tasks.shadow_hand_over_overarm import ShadowHandOverOverarm
from tasks.shadow_hand_catch_over2underarm import ShadowHandCatchOver2Underarm
from tasks.shadow_hand_lift_underarm import ShadowHandLiftUnderarm
from tasks.shadow_hand_bottle_cap import ShadowHandBottleCap

from tasks.shadow_hand_meta_ml1 import ShadowHandMetaML1

from tasks.shadow_hand_catch_overarm_random import ShadowHandCatchOverarmRandom
from tasks.shadow_hand_meta_ml1_random import ShadowHandMetaML1Random
from tasks.two_hand_arms_point2point import TwoHandArmsPoint2Point

from tasks.hand_base.vec_task import VecTaskCPU, VecTaskGPU, VecTaskPython, VecTaskPythonArm
from tasks.hand_base.multi_vec_task import MultiVecTaskPython, SingleVecTaskPythonArm

from utils.config import warn_task_name

import json


def parse_task(args, cfg, cfg_train, sim_params):

    # create native task and pass custom config
    device_id = args.device_id
    rl_device = args.rl_device

    cfg["seed"] = cfg_train.get("seed", -1)
    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]

    if args.task_type == "C++":
        if args.device == "cpu":
            print("C++ CPU")
            task = rlgpu.create_task_cpu(args.task, json.dumps(cfg_task))
            if not task:
                warn_task_name()
            if args.headless:
                task.init(device_id, -1, args.physics_engine, sim_params)
            else:
                task.init(device_id, device_id, args.physics_engine, sim_params)
            env = VecTaskCPU(task, rl_device, False, cfg_train.get("clip_observations", 5.0), cfg_train.get("clip_actions", 1.0))
        else:
            print("C++ GPU")

            task = rlgpu.create_task_gpu(args.task, json.dumps(cfg_task))
            if not task:
                warn_task_name()
            if args.headless:
                task.init(device_id, -1, args.physics_engine, sim_params)
            else:
                task.init(device_id, device_id, args.physics_engine, sim_params)
            env = VecTaskGPU(task, rl_device, cfg_train.get("clip_observations", 5.0), cfg_train.get("clip_actions", 1.0))

    elif args.task_type == "Python":
        print("Python")

        try:
            task = eval(args.task)(
                cfg=cfg,
                sim_params=sim_params,
                physics_engine=args.physics_engine,
                device_type=args.device,
                device_id=device_id,
                headless=args.headless)
        except NameError as e:
            print(e)
            warn_task_name()
        if args.task == "OneFrankaCabinet" :
            env = VecTaskPythonArm(task, rl_device)
        else :
            env = VecTaskPython(task, rl_device)

    
    return task, env
