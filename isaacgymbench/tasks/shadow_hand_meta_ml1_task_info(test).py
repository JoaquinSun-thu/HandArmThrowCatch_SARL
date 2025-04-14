from matplotlib.pyplot import axis
import numpy as np
import os
import random
import torch

from utils.torch_jit_utils import *
from tasks.hand_base.base_task import BaseTask

from isaacgym import gymtorch
from isaacgym import gymapi

# goal_pose_abreast = [-0.39, -1.2, 0.40, 0, 0, 0] # 低处 [-0.39, -1.3, 0.30, 0, 0, 0] ；正常处[-0.39, -1.15, 0.48, 0, 0, 0] ；学弟处[-0.41, -1.15, 0.48,0,0,0] 

def obtrain_task_info(task_name):
    

    if task_name == "catch_overarm_egg":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.05, 0.5+0.48, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "egg"
    if task_name == "catch_overarm_block":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.05, 0.98, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "block"
    if task_name == "catch_overarm_poker":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.03, 0.97, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "poker"
    if task_name == "catch_overarm_banana":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.05, 0.98, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "banana"

    if task_name == "catch_overarm_suger":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.04, 0.98, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "suger" 
    if task_name == "catch_overarm_apple":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [-0.02, 0.04, 0.96, 0, 0, 0]
        goal_pose = [0, -0.6-0.02-0.02, 0.98-0.02-0.04, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "apple"
    if task_name == "catch_overarm_pie":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.04, 0.97, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "pie"
    if task_name == "catch_overarm_mug":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.05, 0.98, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "mug"
    if task_name == "catch_overarm_stapler":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0., 0.03, 0.97, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "stapler"
    if task_name == "catch_overarm_bowl":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.04, 0.97, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "bowl"
    if task_name == "catch_overarm_peach":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.04, 0.96, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "peach"
    if task_name == "catch_overarm_pear":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0.01, 0, 0.96, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "pear"
    if task_name == "catch_overarm_strawberry":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [-0.02, 0.03, 0.97, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "strawberry"
    if task_name == "catch_overarm_pen":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.05, 0.98, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "pen"
    if task_name == "catch_overarm_washer":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.05, 0.98, 0, 0, 0]
        goal_pose =  [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]# [0, -0.6+0.015, 0.98+0.015, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "washer"
    if task_name == "catch_overarm_scissors":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.05, 0.98, 0, 0, 0] # [-0.02, 0.036, 0.98, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "scissors"
    if task_name == "catch_overarm_bottle":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.04, 0.97, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "bottle"
    if task_name == "catch_overarm_bluecup":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.04, 0.97, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "bluecup"
    if task_name == "catch_overarm_plate":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.04, 0.97, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "plate"
    if task_name == "catch_overarm_teabox":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.04, 0.97, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "teabox"
    if task_name == "catch_overarm_clenser":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.04, 0.97, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "clenser"
    if task_name == "catch_overarm_conditioner":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.04, 0.97, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "conditioner"
    if task_name == "catch_overarm_correctionfluid":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.04, 0.97, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "correctionfluid"
    if task_name == "catch_overarm_crackerbox":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.04, 0.97, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "crackerbox"
    if task_name == "catch_overarm_doraemonbowl":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.04, 0.97, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "doraemonbowl"
    if task_name == "catch_overarm_largeclamp":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.04, 0.97, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "largeclamp"
    if task_name == "catch_overarm_flatscrewdrive":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.04, 0.97, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "flatscrewdrive"
    if task_name == "catch_overarm_fork":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.04, 0.97, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "fork"
    if task_name == "catch_overarm_glue":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.04, 0.97, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "glue"
    if task_name == "catch_overarm_liption":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.04, 0.97, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "liption"
    if task_name == "catch_overarm_lemon":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.04, 0.97, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "lemon"
    if task_name == "catch_overarm_orange":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.04, 0.97, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "orange"
    if task_name == "catch_overarm_remotecontroller":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.04, 0.97, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "remotecontroller"
    if task_name == "catch_overarm_sugerbox":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.04, 0.97, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "sugerbox"
    if task_name == "catch_overarm_repellent":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.04, 0.97, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "repellent"
    if task_name == "catch_overarm_shampoo":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0] #pos:x,y,z;euler:z,y,x
        another_hand_start_pose = [0, -0.6, 0.5, 1.57+0.1, 3.14, 3.1415]
        object_pose = [0, 0.04, 0.97, 0, 0, 0]
        goal_pose = [0, -0.6-0.02, 0.98-0.02, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "shampoo"

    underarm_obj_initpose = [0, -0.42, 0.54, 0, 0, 0]
    underarm_obj_goalpose = [ 0, -0.79, 0.50, 0, -0., 0.]
    if task_name == "catch_underarm_poker":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose  # -0.70 0.50; -0.70 0.56;-0.79 0.50
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "poker"
    if task_name == "catch_underarm_banana":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "banana"
    if task_name == "catch_underarm_suger":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "suger"
    if task_name == "catch_underarm_apple":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "apple"
    if task_name == "catch_underarm_pie":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "pie"
    if task_name == "catch_underarm_mug":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "mug"
    if task_name == "catch_underarm_stapler":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "stapler"
    if task_name == "catch_underarm_bowl":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "bowl"
    if task_name == "catch_underarm_peach":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "peach"
    if task_name == "catch_underarm_pear":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "pear"
    if task_name == "catch_underarm_strawberry":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "strawberry"
    if task_name == "catch_underarm_pen":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "pen"
    if task_name == "catch_underarm_washer":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        object_pose[0] = -0.01
        object_pose[3] = 0.
        object_pose[4] = 1.5707
        object_pose[5] = 0
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "washer"
    if task_name == "catch_underarm_scissors":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose # [0, -0.42, 0.54, 0, 0, 0]
        object_pose[0] = 0.02
        object_pose[1] = -0.42
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "scissors"
    if task_name == "catch_underarm_bottle":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "bottle"
    if task_name == "catch_underarm_bluecup":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "bluecup"
    if task_name == "catch_underarm_plate":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "plate"
    if task_name == "catch_underarm_teabox":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "teabox"
    if task_name == "catch_underarm_clenser":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "clenser"
    if task_name == "catch_underarm_conditioner":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "conditioner"
    if task_name == "catch_underarm_correctionfluid":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "correctionfluid"
    if task_name == "catch_underarm_crackerbox":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "crackerbox"
    if task_name == "catch_underarm_doraemonbowl":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "doraemonbowl"
    if task_name == "catch_underarm_largeclamp":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "largeclamp"
    if task_name == "catch_underarm_flatscrewdrive":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "flatscrewdrive"
    if task_name == "catch_underarm_fork":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "fork"
    if task_name == "catch_underarm_glue":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "glue"
    if task_name == "catch_underarm_liption":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "liption"
    if task_name == "catch_underarm_lemon":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "lemon"
    if task_name == "catch_underarm_orange":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "orange"
    if task_name == "catch_underarm_remotecontroller":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "remotecontroller"
    if task_name == "catch_underarm_sugerbox":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "sugerbox"
    if task_name == "catch_underarm_repellent":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "repellent"
    if task_name == "catch_underarm_shampoo":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = underarm_obj_initpose
        goal_pose = underarm_obj_goalpose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "shampoo"

    if task_name == "catch_underarm_0":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = [0, -0.39, 0.54, 0, 0, 0]
        goal_pose = [ 0, -0.79, 0.54, 0, -0., 0.]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "egg"

    if task_name == "catch_underarm_1":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = [0, -0.39, 0.54, 0, 0, 0]
        goal_pose = [0, -0.84, 0.54, 0, -0., 0.]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "egg"

    if task_name == "catch_underarm_2":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = [0, -0.39, 0.54, 0, 0, 0]
        goal_pose = [0.05, -0.79, 0.54, 0., -0., 0.]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "egg"

    if task_name == "catch_underarm_3":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = [0, -1.15, 0.5, 0, 0, 3.1415]
        object_pose = [0, -0.39, 0.54, 0, 0, 0]
        goal_pose = [-0.05, -0.79, 0.54, -0., -0., 0.]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "egg"

    _abreast_goal_pose = [-0.41, -1.15, 0.45, 0, 0, 0]
    _abreast_another_hand_start_pose = [0, -1.12, 0.50, 0, -0.3925, -1.57]
    
    if task_name == "catch_abreast":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.39, -0.55, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "egg"

# 更低处
    # if task_name == "catch_abreast_poker":
    #     hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
    #     another_hand_start_pose = [0, -1.15, 0.5, 0, -0.3925, -1.57]
    #     object_pose = [-0.39, -0.55, 0.54, 0, 0, 0]
    #     goal_pose = [-0.39, -1.3, 0.30, 0, 0, 0]
    #     table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    #     object_asset_options = gymapi.AssetOptions()
    #     object_asset_options.density = 500
    #     object_type = "poker"
    # if task_name == "catch_abreast_banana":
    #     hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
    #     another_hand_start_pose = [0, -1.15, 0.5, 0, -0.3925, -1.57]
    #     object_pose = [-0.39, -0.55, 0.54, 0, 0, 0]
    #     goal_pose = [-0.39, -1.3, 0.30, 0, 0, 0]
    #     table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    #     object_asset_options = gymapi.AssetOptions()
    #     object_asset_options.density = 500
    #     object_type = "banana"
    # if task_name == "catch_abreast_suger":
    #     hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
    #     another_hand_start_pose = [0, -1.15, 0.5, 0, -0.3925, -1.57]
    #     object_pose = [-0.39, -0.55, 0.54, 0, 0, 0]
    #     goal_pose = [-0.39, -1.3, 0.30, 0, 0, 0]
    #     table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    #     object_asset_options = gymapi.AssetOptions()
    #     object_asset_options.density = 500
    #     object_type = "suger"
    # if task_name == "catch_abreast_apple":
    #     hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
    #     another_hand_start_pose = [0, -1.15, 0.5, 0, -0.3925, -1.57]
    #     object_pose = [-0.39, -0.55, 0.54, 0, 0, 0]
    #     goal_pose = [-0.39, -1.3, 0.30, 0, 0, 0]
    #     table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    #     object_asset_options = gymapi.AssetOptions()
    #     object_asset_options.density = 500
    #     object_type = "apple"
    # if task_name == "catch_abreast_pie":
    #     hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
    #     another_hand_start_pose = [0, -1.15, 0.5, 0, -0.3925, -1.57]
    #     object_pose = [-0.39, -0.55, 0.54, 0, 0, 0]
    #     goal_pose = [-0.39, -1.3, 0.30, 0, 0, 0]
    #     table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    #     object_asset_options = gymapi.AssetOptions()
    #     object_asset_options.density = 500
    #     object_type = "pie"
    # if task_name == "catch_abreast_mug":
    #     hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
    #     another_hand_start_pose = [0, -1.15, 0.5, 0, -0.3925, -1.57]
    #     object_pose = [-0.39, -0.55, 0.54, 0, 0, 0]
    #     goal_pose = [-0.39, -1.3, 0.30, 0, 0, 0]
    #     table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    #     object_asset_options = gymapi.AssetOptions()
    #     object_asset_options.density = 500
    #     object_type = "mug"
    # if task_name == "catch_abreast_stapler":
    #     hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
    #     another_hand_start_pose = [0, -1.15, 0.5, 0, -0.3925, -1.57]
    #     object_pose = [-0.39, -0.55, 0.54, 0, 0, 0]
    #     goal_pose = [-0.39, -1.3, 0.30, 0, 0, 0]
    #     table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    #     object_asset_options = gymapi.AssetOptions()
    #     object_asset_options.density = 500
    #     object_type = "stapler"
    # if task_name == "catch_abreast_bowl":
    #     hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
    #     another_hand_start_pose = [0, -1.15, 0.5, 0, -0.3925, -1.57]
    #     object_pose = [-0.39, -0.55, 0.54, 0, 0, 0]
    #     goal_pose = [-0.39, -1.3, 0.30, 0, 0, 0]
    #     table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    #     object_asset_options = gymapi.AssetOptions()
    #     object_asset_options.density = 500
    #     object_type = "bowl"
    # if task_name == "catch_abreast_peach":
    #     hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
    #     another_hand_start_pose = [0, -1.15, 0.5, 0, -0.3925, -1.57]
    #     object_pose = [-0.39, -0.55, 0.54, 0, 0, 0]
    #     goal_pose = [-0.39, -1.3, 0.30, 0, 0, 0]
    #     table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    #     object_asset_options = gymapi.AssetOptions()
    #     object_asset_options.density = 500
    #     object_type = "peach"
    # if task_name == "catch_abreast_pear":
    #     hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
    #     another_hand_start_pose = [0, -1.15, 0.5, 0, -0.3925, -1.57]
    #     object_pose = [-0.39, -0.55, 0.54, 0, 0, 0]
    #     goal_pose = [-0.39, -1.3, 0.30, 0, 0, 0]
    #     table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    #     object_asset_options = gymapi.AssetOptions()
    #     object_asset_options.density = 500
    #     object_type = "pear"
    # if task_name == "catch_abreast_strawberry":
    #     hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
    #     another_hand_start_pose = [0, -1.15, 0.5, 0, -0.3925, -1.57]
    #     object_pose = [-0.39, -0.55, 0.54, 0, 0, 0]
    #     goal_pose = [-0.39, -1.3, 0.30, 0, 0, 0]
    #     table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    #     object_asset_options = gymapi.AssetOptions()
    #     object_asset_options.density = 500
    #     object_type = "strawberry"
    # if task_name == "catch_abreast_pen":
    #     hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
    #     another_hand_start_pose = [0, -1.15, 0.5, 0, -0.3925, -1.57]
    #     object_pose = [-0.39, -0.55, 0.54, 0, 0, 0]
    #     goal_pose = [-0.39, -1.3, 0.30, 0, 0, 0]
    #     table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    #     object_asset_options = gymapi.AssetOptions()
    #     object_asset_options.density = 500
    #     object_type = "pen"
    # if task_name == "catch_abreast_washer":
    #     hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
    #     another_hand_start_pose = [0, -1.15, 0.5, 0, -0.3925, -1.57]
    #     object_pose = [-0.39, -0.55, 0.54, 0, 0, 0]
    #     goal_pose = [-0.39, -1.3, 0.30, 0, 0, 0]
    #     table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    #     object_asset_options = gymapi.AssetOptions()
    #     object_asset_options.density = 500
    #     object_type = "washer"
    # if task_name == "catch_abreast_scissors":
    #     hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
    #     another_hand_start_pose = [0, -1.15, 0.5, 0, -0.3925, -1.57]
    #     object_pose = [-0.39, -0.55, 0.54, 0, 0, 0]
    #     goal_pose = [-0.39, -1.3, 0.30, 0, 0, 0]
    #     table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    #     object_asset_options = gymapi.AssetOptions()
    #     object_asset_options.density = 500
    #     object_type = "scissors"    
# 正常处
    if task_name == "catch_abreast_poker":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "poker"
    if task_name == "catch_abreast_banana":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "banana"
    if task_name == "catch_abreast_suger":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "suger"
    if task_name == "catch_abreast_apple":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "apple"
    if task_name == "catch_abreast_pie":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "pie"
    if task_name == "catch_abreast_mug":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "mug"
    if task_name == "catch_abreast_stapler":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "stapler"
    if task_name == "catch_abreast_bowl":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "bowl"
    if task_name == "catch_abreast_peach":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "peach"
    if task_name == "catch_abreast_pear":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "pear"
    if task_name == "catch_abreast_strawberry":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "strawberry"
    if task_name == "catch_abreast_pen":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "pen"
    if task_name == "catch_abreast_washer":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "washer"
    if task_name == "catch_abreast_scissors":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.58, 0.54, 0, 0, 0] # [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "scissors"
    if task_name == "catch_abreast_bottle":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "bottle"
    if task_name == "catch_abreast_bluecup":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "bluecup"
    if task_name == "catch_abreast_plate":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "plate"
    if task_name == "catch_abreast_teabox":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "teabox"
    if task_name == "catch_abreast_clenser":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "clenser"
    if task_name == "catch_abreast_conditioner":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "conditioner"
    if task_name == "catch_abreast_correctionfluid":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "correctionfluid"
    if task_name == "catch_abreast_crackerbox":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "crackerbox"
    if task_name == "catch_abreast_doraemonbowl":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "doraemonbowl"
    if task_name == "catch_abreast_largeclamp":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "largeclamp"
    if task_name == "catch_abreast_flatscrewdrive":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "flatscrewdrive"
    if task_name == "catch_abreast_fork":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "fork"
    if task_name == "catch_abreast_glue":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "glue"
    if task_name == "catch_abreast_liption":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "liption"
    if task_name == "catch_abreast_lemon":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "lemon"
    if task_name == "catch_abreast_orange":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "orange"
    if task_name == "catch_abreast_remotecontroller":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "remotecontroller"
    if task_name == "catch_abreast_sugerbox":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "sugerbox"
    if task_name == "catch_abreast_repellent":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "repellent"
    if task_name == "catch_abreast_shampoo":
        hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
        another_hand_start_pose = _abreast_another_hand_start_pose
        object_pose = [-0.45, -0.59, 0.54, 0, 0, 0]
        goal_pose = _abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "shampoo"   

    _overarm2abreast_goal_pose = [-0.41, -0.77, 0.45, 0, 0, 0]
    _overarm2abreast_another_hand_start_pose = [0, -0.75, 0.50, 0, -0.3925, -1.57]
    # 正常处
    if task_name == "catch_overarm2abreast_poker":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0]
        another_hand_start_pose = _overarm2abreast_another_hand_start_pose
        object_pose = [0, 0.03, 0.97, 0, 0, 0]
        goal_pose = _overarm2abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "poker"
    if task_name == "catch_overarm2abreast_banana":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0]
        another_hand_start_pose = _overarm2abreast_another_hand_start_pose
        object_pose = [0, 0.05, 0.98, 0, 0, 0]
        goal_pose = _overarm2abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "banana"
    if task_name == "catch_overarm2abreast_suger":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0]
        another_hand_start_pose = _overarm2abreast_another_hand_start_pose
        object_pose = [0, 0.04, 0.98, 0, 0, 0]
        goal_pose = _overarm2abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "suger"
    if task_name == "catch_overarm2abreast_apple":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0]
        another_hand_start_pose = _overarm2abreast_another_hand_start_pose
        object_pose = [0, 0.05, 0.96, 0, 0, 0]
        goal_pose = _overarm2abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "apple"
    if task_name == "catch_overarm2abreast_pie":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0]
        another_hand_start_pose = _overarm2abreast_another_hand_start_pose
        object_pose = [0, 0.04, 0.97, 0, 0, 0]
        goal_pose = _overarm2abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "pie"
    if task_name == "catch_overarm2abreast_mug":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0]
        another_hand_start_pose = _overarm2abreast_another_hand_start_pose
        object_pose = [0, 0.05, 0.98, 0, 0, 0]
        goal_pose = _overarm2abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "mug"
    if task_name == "catch_overarm2abreast_stapler":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0]
        another_hand_start_pose = _overarm2abreast_another_hand_start_pose
        object_pose = [0, 0.03, 0.97, 0, 0, 0]
        goal_pose = [-0.41, -0.77, 0.45, 0, 0, 0]
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "stapler"
    if task_name == "catch_overarm2abreast_bowl":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0]
        another_hand_start_pose = _overarm2abreast_another_hand_start_pose
        object_pose = [0, 0.04, 0.97, 0, 0, 0]
        goal_pose = _overarm2abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "bowl"
    if task_name == "catch_overarm2abreast_peach":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0]
        another_hand_start_pose = _overarm2abreast_another_hand_start_pose
        object_pose = [0, 0.04, 0.96, 0, 0, 0]
        goal_pose = _overarm2abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "peach"
    if task_name == "catch_overarm2abreast_pear":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0]
        another_hand_start_pose = _overarm2abreast_another_hand_start_pose
        object_pose = [0.01, 0, 0.96, 0, 0, 0]
        goal_pose = _overarm2abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "pear"
    if task_name == "catch_overarm2abreast_strawberry":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0]
        another_hand_start_pose = _overarm2abreast_another_hand_start_pose
        object_pose = [0, 0.03, 0.97, 0, 0, 0]
        goal_pose = _overarm2abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "strawberry"
    if task_name == "catch_overarm2abreast_pen":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0]
        another_hand_start_pose = _overarm2abreast_another_hand_start_pose
        object_pose = [0, 0.05, 0.98, 0, 0, 0]
        goal_pose = _overarm2abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "pen"
    if task_name == "catch_overarm2abreast_washer":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0]
        another_hand_start_pose = _overarm2abreast_another_hand_start_pose
        object_pose = [0, 0.05, 0.98, 0, 0, 0]
        goal_pose = _overarm2abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "washer"
    if task_name == "catch_overarm2abreast_scissors":
        hand_start_pose = [0, 0, 0.5, 1.57 +0.1, 3.14, 0]
        another_hand_start_pose = _overarm2abreast_another_hand_start_pose
        object_pose = [0, 0.05, 0.98, 0, 0, 0]
        goal_pose = _overarm2abreast_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "scissors"   

    _under2overarm_goal_pose = [0, -0.95, 0.94+0.01, 0, 0, 0]
    _under2overarm_another_hand_start_pose = [0, -0.9, 0.5, 1.57+0.1, 3.14, 3.1415]

    # 正常处
    if task_name == "catch_under2overarm_poker":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = _under2overarm_another_hand_start_pose
        object_pose = [0-0.01, -0.48, 0.54+0.02, 0, 0, 0]
        goal_pose = _under2overarm_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "poker"
    if task_name == "catch_under2overarm_banana":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = _under2overarm_another_hand_start_pose
        object_pose = [0-0.01, -0.48, 0.54+0.02, 0, 0, 0]
        goal_pose = _under2overarm_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "banana"
    if task_name == "catch_under2overarm_suger":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = _under2overarm_another_hand_start_pose
        object_pose = [0-0.01, -0.48, 0.54+0.02, 1.57, 0, 0]
        goal_pose = _under2overarm_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "suger"
    if task_name == "catch_under2overarm_apple":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = _under2overarm_another_hand_start_pose
        object_pose = [0-0.01, -0.48, 0.54+0.02, 0, 0, 0]
        goal_pose = _under2overarm_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "apple"
    if task_name == "catch_under2overarm_pie":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = _under2overarm_another_hand_start_pose
        object_pose = [0-0.01, -0.48, 0.54+0.02, 0, 0, 0]
        goal_pose = _under2overarm_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "pie"
    if task_name == "catch_under2overarm_mug":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = _under2overarm_another_hand_start_pose
        object_pose = [0-0.01, -0.48, 0.54+0.02, 0, 0, 0]
        goal_pose = _under2overarm_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "mug"
    if task_name == "catch_under2overarm_stapler":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = _under2overarm_another_hand_start_pose
        object_pose = [0-0.01, -0.48, 0.54+0.02, 0, 0, 0]
        goal_pose = _under2overarm_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "stapler"
    if task_name == "catch_under2overarm_bowl":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = _under2overarm_another_hand_start_pose
        object_pose = [0-0.01, -0.48, 0.54+0.02, 0, 0, 0]
        goal_pose = _under2overarm_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "bowl"
    if task_name == "catch_under2overarm_peach":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = _under2overarm_another_hand_start_pose
        object_pose = [0-0.01, -0.48, 0.54+0.02, 0, 0, 0]
        goal_pose = _under2overarm_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "peach"
    if task_name == "catch_under2overarm_pear":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = _under2overarm_another_hand_start_pose
        object_pose = [0-0.01, -0.48, 0.54+0.02, 0, 0, 0]
        goal_pose = _under2overarm_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "pear"
    if task_name == "catch_under2overarm_strawberry":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = _under2overarm_another_hand_start_pose
        object_pose = [0-0.01, -0.48, 0.54, 0, 0, 0]
        goal_pose = _under2overarm_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "strawberry"
    if task_name == "catch_under2overarm_pen":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = _under2overarm_another_hand_start_pose
        object_pose = [0-0.01, -0.48, 0.54+0.02, 0, 0, 0]
        goal_pose = _under2overarm_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "pen"
    if task_name == "catch_under2overarm_washer":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = _under2overarm_another_hand_start_pose
        object_pose = [0-0.01, -0.48, 0.54+0.02, 0, 0, 0]
        goal_pose = _under2overarm_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "washer"
    if task_name == "catch_under2overarm_scissors":
        hand_start_pose = [0, 0, 0.5, 0, 0, 0]
        another_hand_start_pose = _under2overarm_another_hand_start_pose
        object_pose = [0-0.01, -0.48, 0.54+0.02, 0, 0, 0]
        goal_pose = _under2overarm_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "scissors"
    
    _overarmout45_hand_start_pose = [0, 0, 0.5, 1.57+0.1+0.775, 3.14, 0]
    _overarmout45_another_hand_start_pose = [0, -0.3, 0.5, 1.57+0.1+0.775, 3.14, 3.1415]   # zyx
    _overarmout45_object_pose = [0-0.01, 0.04+0.3, 0.94+0.02-0.13, 0, 0, 0]
    _overarmout45_goal_pose = [0, -0.3-0.02-0.3, 0.94+0.01-0.13, 0, 0, 0]
    
    # 正常处
    if task_name == "catch_overarmout45_poker":
        hand_start_pose = _overarmout45_hand_start_pose
        another_hand_start_pose = _overarmout45_another_hand_start_pose
        object_pose = _overarmout45_object_pose
        goal_pose = _overarmout45_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "poker"
    if task_name == "catch_overarmout45_banana":
        hand_start_pose = _overarmout45_hand_start_pose
        another_hand_start_pose = _overarmout45_another_hand_start_pose
        object_pose = _overarmout45_object_pose
        goal_pose = _overarmout45_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "banana"
    if task_name == "catch_overarmout45_suger":
        hand_start_pose = _overarmout45_hand_start_pose
        another_hand_start_pose = _overarmout45_another_hand_start_pose
        object_pose = _overarmout45_object_pose
        goal_pose = _overarmout45_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "suger"
    if task_name == "catch_overarmout45_apple":
        hand_start_pose = _overarmout45_hand_start_pose
        another_hand_start_pose = _overarmout45_another_hand_start_pose
        object_pose = _overarmout45_object_pose
        goal_pose = _overarmout45_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "apple"
    if task_name == "catch_overarmout45_pie":
        hand_start_pose = _overarmout45_hand_start_pose
        another_hand_start_pose = _overarmout45_another_hand_start_pose
        object_pose = _overarmout45_object_pose
        goal_pose = _overarmout45_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "pie"
    if task_name == "catch_overarmout45_mug":
        hand_start_pose = _overarmout45_hand_start_pose
        another_hand_start_pose = _overarmout45_another_hand_start_pose
        object_pose = _overarmout45_object_pose
        goal_pose = _overarmout45_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "mug"
    if task_name == "catch_overarmout45_stapler":
        hand_start_pose = _overarmout45_hand_start_pose
        another_hand_start_pose = _overarmout45_another_hand_start_pose
        object_pose = _overarmout45_object_pose
        goal_pose = _overarmout45_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "stapler"
    if task_name == "catch_overarmout45_bowl":
        hand_start_pose = _overarmout45_hand_start_pose
        another_hand_start_pose = _overarmout45_another_hand_start_pose
        object_pose = _overarmout45_object_pose
        goal_pose = _overarmout45_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "bowl"
    if task_name == "catch_overarmout45_peach":
        hand_start_pose = _overarmout45_hand_start_pose
        another_hand_start_pose = _overarmout45_another_hand_start_pose
        object_pose = _overarmout45_object_pose
        goal_pose = _overarmout45_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "peach"
    if task_name == "catch_overarmout45_pear":
        hand_start_pose = _overarmout45_hand_start_pose
        another_hand_start_pose = _overarmout45_another_hand_start_pose
        object_pose = _overarmout45_object_pose
        goal_pose = _overarmout45_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "pear"
    if task_name == "catch_overarmout45_strawberry":
        hand_start_pose = _overarmout45_hand_start_pose
        another_hand_start_pose = _overarmout45_another_hand_start_pose
        object_pose = _overarmout45_object_pose
        goal_pose = _overarmout45_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "strawberry"
    if task_name == "catch_overarmout45_pen":
        hand_start_pose = _overarmout45_hand_start_pose
        another_hand_start_pose = _overarmout45_another_hand_start_pose
        object_pose = _overarmout45_object_pose
        goal_pose = _overarmout45_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "pen"
    if task_name == "catch_overarmout45_washer":
        hand_start_pose = _overarmout45_hand_start_pose
        another_hand_start_pose = _overarmout45_another_hand_start_pose
        object_pose = _overarmout45_object_pose
        goal_pose = _overarmout45_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "washer"
    if task_name == "catch_overarmout45_scissors":
        hand_start_pose = _overarmout45_hand_start_pose
        another_hand_start_pose = _overarmout45_another_hand_start_pose
        object_pose = _overarmout45_object_pose
        goal_pose = _overarmout45_goal_pose
        table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_type = "scissors"    
       

    if task_name == "lift_pot":
        hand_start_pose = [0, 0.05, 0.45, 0, 0, 0]
        another_hand_start_pose = [0, -1.25, 0.45, 0, 0, 3.14159]
        object_pose = [0, -0.6, 0.45, 0, 0, 0]
        goal_pose = [0, -0.39, 1, 0, 0, 0]
        table_pose_dim = [0.0, -0.6, 0.5 * 0.4, 0, 0, 0, 0.3, 0.3, 0.4]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 1000
        object_type = "pot"

    if task_name == "door_open_outward":
        hand_start_pose = [0.55, 0.2, 0.6, 3.14159, 1.57, 1.57]
        another_hand_start_pose = [0.55, -0.2, 0.6, 3.14159, -1.57, 1.57]
        object_pose = [0.0, 0., 0.7, 0, 0.0, 0.0]
        goal_pose = [0, -0.39, 10, 0, 0, 0]                
        table_pose_dim = [0.0, -0.6, 0, 0, 0, 0, 0.3, 0.3, 0.]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_asset_options.fix_base_link = True
        object_asset_options.disable_gravity = True
        object_asset_options.use_mesh_materials = True
        object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        object_asset_options.override_com = True
        object_asset_options.override_inertia = True
        object_asset_options.vhacd_enabled = True
        object_asset_options.vhacd_params = gymapi.VhacdParams()
        object_asset_options.vhacd_params.resolution = 200000
        object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        object_type = "door"

    if task_name == "door_close_inward":
        hand_start_pose = [0.55, 0.2, 0.6, 3.14159, 1.57, 1.57]
        another_hand_start_pose = [0.55, -0.2, 0.6, 3.14159, -1.57, 1.57]
        object_pose = [0.0, 0., 0.7, 0, 3.14159, 0.0]
        goal_pose = [0, -0.39, 10, 0, 0, 0]                
        table_pose_dim = [0.0, -0.6, 0, 0, 0, 0, 0.3, 0.3, 0.]
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_asset_options.fix_base_link = True
        object_asset_options.disable_gravity = True
        object_asset_options.use_mesh_materials = True
        object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        object_asset_options.override_com = True
        object_asset_options.override_inertia = True
        object_asset_options.vhacd_enabled = True
        object_asset_options.vhacd_params = gymapi.VhacdParams()
        object_asset_options.vhacd_params.resolution = 100000
        object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        object_type = "door"

    return hand_start_pose, another_hand_start_pose, object_pose, goal_pose, table_pose_dim, object_asset_options, object_type

@torch.jit.script
def compute_hand_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot, object_left_handle_pos, object_right_handle_pos, left_hand_base_pos, right_hand_base_pos,
    left_hand_pos, right_hand_pos, right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_lf_pos, right_hand_th_pos,
    left_hand_ff_pos, left_hand_mf_pos, left_hand_rf_pos, left_hand_lf_pos, left_hand_th_pos,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, this_task: str, device: str = 'cuda:0'
): # 临时添加left_hand_base_pos, right_hand_base_pos,（catch_overarm里面的参数）
    # Distance from the hand to the object
    if this_task in ["catch_underarm", "hand_over", "catch_abreast", "catch_over2underarm"]:
        goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)

        # Orientation alignment for the cube in hand and goal cube
        quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
        rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

        dist_rew = goal_dist
        # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

        action_penalty = torch.sum(actions ** 2, dim=-1)

        # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
        reward = torch.exp(-0.2*(dist_rew * dist_reward_scale + rot_dist))

        # Find out which envs hit the goal and update successes count
        goal_resets = torch.where(torch.abs(goal_dist) <= 0.03, torch.ones_like(reset_goal_buf), reset_goal_buf)
        successes = successes + goal_resets

        # Success bonus: orientation is within `success_tolerance` of goal orientation
        reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

        # Fall penalty: distance to the goal is larger than a threashold
        reward = torch.where(object_pos[:, 2] <= 0.2, reward + fall_penalty, reward)

        # Check env termination conditions, including maximum success number
        resets = torch.where(object_pos[:, 2] <= 0.2, torch.ones_like(reset_buf), reset_buf)
        if max_consecutive_successes > 0:
            # Reset progress buffer on goal envs if max_consecutive_successes > 0
            progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
            resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
        resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

        # Apply penalty for not reaching the goal
        if max_consecutive_successes > 0:
            reward = torch.where(progress_buf >= max_episode_length, reward + 0.5 * fall_penalty, reward)

        num_resets = torch.sum(resets)
        finished_cons_successes = torch.sum(successes * resets.float())

        cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

        return reward, resets, goal_resets, progress_buf, successes, cons_successes
    # add this catch_overarm
    if this_task in ["catch_overarm"]:
        # Distance from the hand to the object
        goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
        # if ignore_z_rot:
        #     success_tolerance = 2.0 * success_tolerance

        # Orientation alignment for the cube in hand and goal cube
        quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
        rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

        dist_rew = goal_dist
        # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

        action_penalty = torch.sum(actions ** 2, dim=-1)

        # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
        reward = torch.exp(-0.1*(dist_rew * dist_reward_scale + rot_dist))

        # Find out which envs hit the goal and update successes count
        goal_resets = torch.where(torch.abs(goal_dist) <= 0, torch.ones_like(reset_goal_buf), reset_goal_buf)
        successes = successes + goal_resets

        # Success bonus: orientation is within `success_tolerance` of goal orientation
        reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

        # Fall penalty: distance to the goal is larger than a threashold
        reward = torch.where(object_pos[:, 2] <= 0.8, reward + fall_penalty, reward)

        # Check env termination conditions, including maximum success number
        right_hand_base_dist = torch.norm(right_hand_base_pos - torch.tensor([0.0, 0.0, 0.5], dtype=torch.float, device=device), p=2, dim=-1)
    
        # small objects
        left_hand_base_dist = torch.norm(left_hand_base_pos - torch.tensor([0.0, -0.6, 0.5], dtype=torch.float, device=device), p=2, dim=-1)  # -0.8
        # big objects
        # left_hand_base_dist = torch.norm(left_hand_base_pos - torch.tensor([0.0, -0.6, 0.5], dtype=torch.float, device="cuda:0"), p=2, dim=-1)  # -0.8

        resets = torch.where(right_hand_base_dist >= 0.1, torch.ones_like(reset_buf), reset_buf)
        resets = torch.where(left_hand_base_dist >= 0.1, torch.ones_like(resets), resets)
        resets = torch.where(object_pos[:, 2] <= 0.8, torch.ones_like(resets), resets)
        if max_consecutive_successes > 0:
            # Reset progress buffer on goal envs if max_consecutive_successes > 0
            progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
            resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
        resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

        # Apply penalty for not reaching the goal
        if max_consecutive_successes > 0:
            reward = torch.where(progress_buf >= max_episode_length, reward + 0.5 * fall_penalty, reward)

        num_resets = torch.sum(resets)
        finished_cons_successes = torch.sum(successes * resets.float())

        cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

        return reward, resets, goal_resets, progress_buf, successes, cons_successes

    if this_task in ["door_open_outward"]:
        # Distance from the hand to the object
        goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
        # goal_dist = target_pos[:, 2] - object_pos[:, 2]

        right_hand_dist = torch.norm(object_right_handle_pos - right_hand_pos, p=2, dim=-1)
        left_hand_dist = torch.norm(object_left_handle_pos - left_hand_pos, p=2, dim=-1)

        right_hand_finger_dist = (torch.norm(object_right_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(object_right_handle_pos - right_hand_mf_pos, p=2, dim=-1)
                                + torch.norm(object_right_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(object_right_handle_pos - right_hand_lf_pos, p=2, dim=-1) 
                                + torch.norm(object_right_handle_pos - right_hand_th_pos, p=2, dim=-1))
        left_hand_finger_dist = (torch.norm(object_left_handle_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(object_left_handle_pos - left_hand_mf_pos, p=2, dim=-1)
                                + torch.norm(object_left_handle_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(object_left_handle_pos - left_hand_lf_pos, p=2, dim=-1) 
                                + torch.norm(object_left_handle_pos - left_hand_th_pos, p=2, dim=-1))
        

        # Orientation alignment for the cube in hand and goal cube
        # quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
        # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

        right_hand_dist_rew = right_hand_finger_dist
        left_hand_dist_rew = left_hand_finger_dist

        # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

        action_penalty = torch.sum(actions ** 2, dim=-1)

        # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
        # reward = torch.exp(-0.05*(up_rew * dist_reward_scale)) + torch.exp(-0.05*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.05*(left_hand_dist_rew * dist_reward_scale))
        up_rew = torch.zeros_like(right_hand_dist_rew)
        up_rew = torch.where(right_hand_finger_dist < 0.5,
                        torch.where(left_hand_finger_dist < 0.5,
                                        torch.abs(object_right_handle_pos[:, 1] - object_left_handle_pos[:, 1]) * 2, up_rew), up_rew)
        # up_rew =  torch.where(right_hand_finger_dist <= 0.3, torch.norm(bottle_cap_up - bottle_pos, p=2, dim=-1) * 30, up_rew)

        # reward = torch.exp(-0.1*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.1*(left_hand_dist_rew * dist_reward_scale))
        reward = 2 - right_hand_dist_rew - left_hand_dist_rew + up_rew

        resets = torch.where(right_hand_finger_dist >= 1.5, torch.ones_like(reset_buf), reset_buf)
        resets = torch.where(left_hand_finger_dist >= 1.5, torch.ones_like(resets), resets)
        # resets = torch.where(left_hand_dist >= 0.2, torch.ones_like(resets), resets)

        # print(right_hand_dist_rew[0])
        # print(left_hand_dist_rew[0])
        # print(up_rew[0])

        # Find out which envs hit the goal and update successes count
        resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

        goal_resets = torch.zeros_like(resets)

        num_resets = torch.sum(resets)
        finished_cons_successes = torch.sum(successes * resets.float())

        cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

        return reward, resets, goal_resets, progress_buf, successes, cons_successes

    if this_task in ["lift_pot"]:
        # Distance from the hand to the object
        goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
        # goal_dist = target_pos[:, 2] - object_pos[:, 2]
        right_hand_dist = torch.norm(object_right_handle_pos - right_hand_pos, p=2, dim=-1)
        left_hand_dist = torch.norm(object_right_handle_pos - left_hand_pos, p=2, dim=-1)
        # Orientation alignment for the cube in hand and goal cube
        # quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
        # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

        right_hand_dist_rew = right_hand_dist
        left_hand_dist_rew = left_hand_dist

        # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

        action_penalty = torch.sum(actions ** 2, dim=-1)

        # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
        # reward = torch.exp(-0.05*(up_rew * dist_reward_scale)) + torch.exp(-0.05*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.05*(left_hand_dist_rew * dist_reward_scale))
        up_rew = torch.zeros_like(right_hand_dist_rew)
        up_rew = torch.where(right_hand_dist < 0.08,
                            torch.where(left_hand_dist < 0.08,
                                            3*(0.985 - goal_dist), up_rew), up_rew)
        
        reward = 0.2 - right_hand_dist_rew - left_hand_dist_rew + up_rew

        resets = torch.where(object_pos[:, 2] <= 0.3, torch.ones_like(reset_buf), reset_buf)
        resets = torch.where(right_hand_dist >= 0.2, torch.ones_like(resets), resets)
        resets = torch.where(left_hand_dist >= 0.2, torch.ones_like(resets), resets)

        # Find out which envs hit the goal and update successes count
        resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

        goal_resets = torch.zeros_like(resets)

        num_resets = torch.sum(resets)
        finished_cons_successes = torch.sum(successes * resets.float())

        cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

        return reward, resets, goal_resets, progress_buf, successes, cons_successes

@torch.jit.script
def compute_hand_reward_catch_overarm(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot, left_hand_base_pos, right_hand_base_pos,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, device: str
): #, ignore_z_rot: bool
    # Distance from the hand to the object
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    # if ignore_z_rot:
    #     success_tolerance = 2.0 * success_tolerance

    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    dist_rew = goal_dist
    # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = torch.exp(-0.1*(dist_rew * dist_reward_scale + rot_dist))

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(goal_dist) <= 0, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threashold
    reward = torch.where(object_pos[:, 2] <= 0.8, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    right_hand_base_dist = torch.norm(right_hand_base_pos - torch.tensor([0.0, 0.0, 0.5], dtype=torch.float, device=device), p=2, dim=-1)
    
    # small objects
    left_hand_base_dist = torch.norm(left_hand_base_pos - torch.tensor([0.0, -0.6, 0.5], dtype=torch.float, device=device), p=2, dim=-1)  # -0.8
    # big objects
    # left_hand_base_dist = torch.norm(left_hand_base_pos - torch.tensor([0.0, -0.6, 0.5], dtype=torch.float, device="cuda:0"), p=2, dim=-1)  # -0.8

    goal_dist_x = torch.abs(target_pos[:, 0] - object_pos[:, 0])
    goal_dist_y = object_pos[:, 1] - target_pos[:, 1]

    resets = torch.where(right_hand_base_dist >= 0.1, torch.ones_like(reset_buf), reset_buf)
    #resets = torch.where(left_hand_base_dist >= 0.1, torch.ones_like(resets), resets)
    # Z
    resets = torch.where(object_pos[:, 2] <= 0.8, torch.ones_like(resets), resets)
    resets = torch.where(object_pos[:, 2] >= 1.5, torch.ones_like(resets), resets)
    # X
    resets = torch.where(goal_dist_x >= 0.2, torch.ones_like(resets), resets)
    # Y
    resets = torch.where(object_pos[:, 1] <= -0.9, torch.ones_like(resets), resets)
    resets = torch.where(object_pos[:, 1] >= 0.1, torch.ones_like(resets), resets)

    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(progress_buf >= max_episode_length, reward + 0.5 * fall_penalty, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, cons_successes

@torch.jit.script
def compute_hand_reward_catch_underarm(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot, left_hand_base_pos, right_hand_base_pos,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, device: str
):#, ignore_z_rot: bool
    # Distance from the hand to the object
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    # goal_x_dist = torch.norm(target_pos[:, 0] - object_pos[:, 0], p=2, dim=-1, keepdim=True)

    # if ignore_z_rot:
    #     success_tolerance = 2.0 * success_tolerance

    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    dist_rew = goal_dist
    # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = torch.exp(-0.2*(dist_rew * dist_reward_scale + rot_dist))
    # reward = torch.exp(-0.2 * (dist_rew + rot_dist * rot_reward_scale))

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(goal_dist) <= 0, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets
    # successes = torch.where(successes == 0, 
    #                 torch.where(goal_dist < 0.03, torch.ones_like(successes), successes), successes)

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threashold
    reward = torch.where(object_pos[:, 2] <= 0.1, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    right_hand_base_dist = torch.norm(right_hand_base_pos - torch.tensor([0.0, 0.0, 0.5], dtype=torch.float, device=device), p=2, dim=-1)
    
    # small objects
    left_hand_base_dist = torch.norm(left_hand_base_pos - torch.tensor([0.0, -1.15, 0.5], dtype=torch.float, device=device), p=2, dim=-1)  # -0.8
    # big objects
    # left_hand_base_dist = torch.norm(left_hand_base_pos - torch.tensor([0.0, -0.6, 0.5], dtype=torch.float, device="cuda:0"), p=2, dim=-1)  # -0.8
    # Fall penalty x, y and z: distance to the goal is larger than a threashold
    goal_dist_x = torch.abs(target_pos[:, 0] - object_pos[:, 0])
    goal_dist_y = object_pos[:, 1] - target_pos[:, 1]

    resets = torch.where(right_hand_base_dist >= 0.2, torch.ones_like(reset_buf), reset_buf)
    resets = torch.where(left_hand_base_dist >= 0.2, torch.ones_like(resets), resets)
    # Z 
    resets = torch.where(object_pos[:, 2] <= 0.1, torch.ones_like(resets), resets)
    resets = torch.where(object_pos[:, 2] >= 1.5, torch.ones_like(resets), resets)
    # X
    resets = torch.where(goal_dist_x >= 0.3, torch.ones_like(resets), resets)
    # Y
    resets = torch.where(object_pos[:, 1] <= -1.2, torch.ones_like(resets), resets)
    resets = torch.where(object_pos[:, 1] >= -0.35, torch.ones_like(resets), resets)
    # Fall penalty New:
    # resets_near = torch.where(goal_dist_y <= 0.15, torch.ones_like(resets), resets)
    # resets_near_fall = torch.where(object_pos[:, 2] <= 0.8, torch.ones_like(resets), resets)
    # resets = resets_near * resets_near_fall
    


    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(progress_buf >= max_episode_length, reward + 0.5 * fall_penalty, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, cons_successes

@torch.jit.script
def compute_hand_reward_catch_abreast(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot, left_hand_pos, right_hand_pos, left_hand_base_pos, right_hand_base_pos,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, device: str
): #, ignore_z_rot: bool
    # Distance from the hand to the object
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    # goal_x_dist = torch.norm(target_pos[:, 0] - object_pos[:, 0], p=2, dim=-1, keepdim=True)

    # if ignore_z_rot:
    #     success_tolerance = 2.0 * success_tolerance

    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    dist_rew = goal_dist
    # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = torch.exp(-0.2*(dist_rew * dist_reward_scale + rot_dist))
    # reward = torch.exp(-0.2 * (dist_rew + rot_dist * rot_reward_scale))

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(goal_dist) <= 0, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threashold
    reward = torch.where(object_pos[:, 2] <= 0.2, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    # resets = torch.where(right_hand_pos[:, 1] <= -0.7, torch.ones_like(reset_buf), reset_buf)
    # resets = torch.where(right_hand_pos[:, 2] >= 0.7, torch.ones_like(resets), resets)
    right_hand_base_dist = torch.norm(right_hand_base_pos - torch.tensor([-0.3, -0.55, 0.5], dtype=torch.float, device=device), p=2, dim=-1)
    left_hand_base_dist = torch.norm(left_hand_base_pos - torch.tensor([-0.3, -1.15, 0.5], dtype=torch.float, device=device), p=2, dim=-1)

    resets = torch.where(right_hand_base_dist >= 0.1, torch.ones_like(reset_buf), reset_buf)
    #resets = torch.where(left_hand_base_dist >= 0.1, torch.ones_like(resets), resets)

    resets = torch.where(object_pos[:, 2] <= 0.2, torch.ones_like(resets), resets)
    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(progress_buf >= max_episode_length, reward + 0.5 * fall_penalty, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, cons_successes

@torch.jit.script
def compute_hand_reward_catch_under2overarm(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot, left_hand_base_pos, right_hand_base_pos,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, device: str
): #, ignore_z_rot: bool
    # Distance from the hand to the object
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    # if ignore_z_rot:
    #     success_tolerance = 2.0 * success_tolerance

    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    dist_rew = goal_dist
    # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = torch.exp(-0.1*(dist_rew * dist_reward_scale + rot_dist))

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(goal_dist) <= 0, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threashold
    reward = torch.where(object_pos[:, 2] <= 0.2, reward + fall_penalty, reward)
    
    
    # Check env termination conditions, including maximum success number
    right_hand_base_dist = torch.norm(right_hand_base_pos - torch.tensor([0.0, 0.0, 0.5], dtype=torch.float, device=device), p=2, dim=-1)
    
    # small objects
    left_hand_base_dist = torch.norm(left_hand_base_pos - torch.tensor([0.0, -0.9, 0.5], dtype=torch.float, device=device), p=2, dim=-1)  # -0.8
    # big objects
    # left_hand_base_dist = torch.norm(left_hand_base_pos - torch.tensor([0.0, -0.6, 0.5], dtype=torch.float, device="cuda:0"), p=2, dim=-1)  # -0.8
    # Fall penalty x, y and z: distance to the goal is larger than a threashold
    goal_dist_x = torch.abs(target_pos[:, 0] - object_pos[:, 0])
    goal_dist_y = object_pos[:, 1] - target_pos[:, 1]

    resets = torch.where(right_hand_base_dist >= 0.4, torch.ones_like(reset_buf), reset_buf)
    #resets = torch.where(left_hand_base_dist >= 0.1, torch.ones_like(resets), resets)
    # Z 
    resets = torch.where(object_pos[:, 2] <= 0.2, torch.ones_like(resets), resets)
    resets = torch.where(object_pos[:, 2] >= 1.5, torch.ones_like(resets), resets)
    # X
    resets = torch.where(goal_dist_x >= 0.2, torch.ones_like(resets), resets)
    # Y
    resets = torch.where(object_pos[:, 1] <= -1.2, torch.ones_like(resets), resets)
    resets = torch.where(object_pos[:, 1] >= -0.35, torch.ones_like(resets), resets)
    # Fall penalty New:
    resets_near = torch.where(goal_dist_y <= 0.15, torch.ones_like(resets), resets)
    resets_near_fall = torch.where(object_pos[:, 2] <= 0.8, torch.ones_like(resets), resets)
    resets = resets_near * resets_near_fall
    
    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(progress_buf >= max_episode_length, reward + 0.5 * fall_penalty, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, cons_successes

@torch.jit.script
def compute_hand_reward_catch_overarm2abreast(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot, left_hand_base_pos, right_hand_base_pos,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, device: str
):
    # Distance from the hand to the object
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    # if ignore_z_rot:
    #     success_tolerance = 2.0 * success_tolerance

    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    dist_rew = goal_dist
    # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = torch.exp(-0.2*(dist_rew * dist_reward_scale + rot_dist))

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(goal_dist) <= 0, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Check env termination conditions, including maximum success number
    right_hand_base_dist = torch.norm(right_hand_base_pos - torch.tensor([0.0, 0.0, 0.5], dtype=torch.float, device=device), p=2, dim=-1)
    left_hand_base_dist = torch.norm(left_hand_base_pos - torch.tensor([-0.001, -0.75, 0.50], dtype=torch.float, device=device), p=2, dim=-1)

    resets = torch.where(right_hand_base_dist >= 0.1, torch.ones_like(reset_buf), reset_buf)
    #resets = torch.where(left_hand_base_dist >= 0.1, torch.ones_like(resets), resets)
    resets = torch.where(object_pos[:, 2] <= 0.3, torch.ones_like(resets), resets)
    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(progress_buf >= max_episode_length, reward + 0.5 * fall_penalty, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, cons_successes


@torch.jit.script
def compute_hand_reward_catch_overarmout45(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot, left_hand_base_pos, right_hand_base_pos,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, device: str
): #, ignore_z_rot: bool
    # Distance from the hand to the object
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    # if ignore_z_rot:
    #     success_tolerance = 2.0 * success_tolerance

    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    dist_rew = goal_dist
    # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = torch.exp(-0.1*(dist_rew * dist_reward_scale + rot_dist))

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(goal_dist) <= 0.01, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threashold
    reward = torch.where(object_pos[:, 2] <= 0.6, reward + fall_penalty, reward)
    
    
    # Check env termination conditions, including maximum success number
    right_hand_base_dist = torch.norm(right_hand_base_pos - torch.tensor([0.0, 0.0, 0.5], dtype=torch.float, device=device), p=2, dim=-1)
    
    # small objects
    left_hand_base_dist = torch.norm(left_hand_base_pos - torch.tensor([0.0, -0.3, 0.5], dtype=torch.float, device=device), p=2, dim=-1)  # -0.8
    # big objects
    # left_hand_base_dist = torch.norm(left_hand_base_pos - torch.tensor([0.0, -0.6, 0.5], dtype=torch.float, device="cuda:0"), p=2, dim=-1)  # -0.8
    # Fall penalty x, y and z: distance to the goal is larger than a threashold
    goal_dist_x = torch.abs(target_pos[:, 0] - object_pos[:, 0])
    goal_dist_y = object_pos[:, 1] - target_pos[:, 1]

    resets = torch.where(right_hand_base_dist >= 0.4, torch.ones_like(reset_buf), reset_buf)
    resets = torch.where(left_hand_base_dist >= 0.4, torch.ones_like(resets), resets)
    # Z 
    resets = torch.where(object_pos[:, 2] <= 0.6, torch.ones_like(resets), resets)
    resets = torch.where(object_pos[:, 2] >= 1.4, torch.ones_like(resets), resets)
    # X
    resets = torch.where(goal_dist_x >= 0.2, torch.ones_like(resets), resets)
    # Y
    resets = torch.where(object_pos[:, 1] <= -1.2, torch.ones_like(resets), resets)
    resets = torch.where(object_pos[:, 1] >= 0.4, torch.ones_like(resets), resets)
    # # Fall penalty New:
    # resets_near = torch.where(goal_dist_y <= 0.15, torch.ones_like(resets), resets)
    # resets_near_fall = torch.where(object_pos[:, 2] <= 0.8, torch.ones_like(resets), resets)
    # resets = resets_near * resets_near_fall
    
    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(progress_buf >= max_episode_length, reward + 0.5 * fall_penalty, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, cons_successes

# from matplotlib.pyplot import axis
# import numpy as np
# import os
# import random
# import torch

# from bidexhands.utils.torch_jit_utils import *
# from bidexhands.tasks.hand_base.base_task import BaseTask

# from isaacgym import gymtorch
# from isaacgym import gymapi

# def obtrain_task_info(task_name):
#     if task_name == "catch_underarm_0":
#         hand_start_pose = [0, 0, 0.5, 0, 0, 0]
#         another_hand_start_pose = [0, -1, 0.5, 0, 0, 3.1415]
#         object_pose = [0, -0.39, 0.56, 0, 0, 0]
#         goal_pose = [ 0, -0.64, 0.54, 0, -0., 0.]
#         table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
#         object_asset_options = gymapi.AssetOptions()
#         object_asset_options.density = 500
#         object_type = "egg"

#     if task_name == "catch_underarm_1":
#         hand_start_pose = [0, 0, 0.5, 0, 0, 0]
#         another_hand_start_pose = [0, -1, 0.5, 0, 0, 3.1415]
#         object_pose = [0, -0.39, 0.56, 0, 0, 0]
#         goal_pose = [0, -0.61, 0.54, 0, -0., 0.]
#         table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
#         object_asset_options = gymapi.AssetOptions()
#         object_asset_options.density = 500
#         object_type = "egg"

#     if task_name == "catch_underarm_2":
#         hand_start_pose = [0, 0, 0.5, 0, 0, 0]
#         another_hand_start_pose = [0, -1, 0.5, 0, 0, 3.1415]
#         object_pose = [0, -0.39, 0.56, 0, 0, 0]
#         goal_pose = [0, -0.67, 0.54, 0, -0., 0.]
#         table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
#         object_asset_options = gymapi.AssetOptions()
#         object_asset_options.density = 500
#         object_type = "egg"

#     if task_name == "catch_underarm_3":
#         hand_start_pose = [0, 0, 0.5, 0, 0, 0]
#         another_hand_start_pose = [0, -1, 0.5, 0, 0, 3.1415]
#         object_pose = [0, -0.39, 0.56, 0, 0, 0]
#         goal_pose = [0.03, -0.64, 0.54, 0., -0., 0.]
#         table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
#         object_asset_options = gymapi.AssetOptions()
#         object_asset_options.density = 500
#         object_type = "egg"

#     if task_name == "catch_underarm_4":
#         hand_start_pose = [0, 0, 0.5, 0, 0, 0]
#         another_hand_start_pose = [0, -1., 0.5, 0, 0, 3.1415]
#         object_pose = [0, -0.39, 0.56, 0, 0, 0]
#         goal_pose = [-0.03, -0.64, 0.54, -0., -0., 0.]
#         table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
#         object_asset_options = gymapi.AssetOptions()
#         object_asset_options.density = 500
#         object_type = "egg"

#     if task_name == "catch_underarm_5":
#         hand_start_pose = [0, 0, 0.5, 0, 0, 0]
#         another_hand_start_pose = [0, -1, 0.5, 0, 0, 3.1415]
#         object_pose = [0, -0.39, 0.56, 0, 0, 0]
#         goal_pose = [0, -0.64, 0.51, 0, -0., 0.]
#         table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
#         object_asset_options = gymapi.AssetOptions()
#         object_asset_options.density = 500
#         object_type = "egg"

#     if task_name == "catch_abreast":
#         hand_start_pose = [0, -0.55, 0.5, 0, 0.3925, -1.57]
#         another_hand_start_pose = [0, -1.15, 0.5, 0, -0.3925, -1.57]
#         object_pose = [-0.39, -0.55, 0.54, 0, 0, 0]
#         goal_pose = [-0.39, -0.55, 0.54, 0, 0, 0]
#         table_pose_dim = [0, 0, 0, 0, 0, 0, 0, 0, 0]
#         object_asset_options = gymapi.AssetOptions()
#         object_asset_options.density = 500
#         object_type = "egg"

#     if task_name == "lift_pot":
#         hand_start_pose = [0, 0.05, 0.45, 0, 0, 0]
#         another_hand_start_pose = [0, -1.25, 0.45, 0, 0, 3.14159]
#         object_pose = [0, -0.6, 0.45, 0, 0, 0]
#         goal_pose = [0, -0.39, 1, 0, 0, 0]
#         table_pose_dim = [0.0, -0.6, 0.5 * 0.4, 0, 0, 0, 0.3, 0.3, 0.4]
#         object_asset_options = gymapi.AssetOptions()
#         object_asset_options.density = 1000
#         object_type = "pot"

#     if task_name == "door_open_outward":
#         hand_start_pose = [0.55, 0.2, 0.6, 3.14159, 1.57, 1.57]
#         another_hand_start_pose = [0.55, -0.2, 0.6, 3.14159, -1.57, 1.57]
#         object_pose = [0.0, 0., 0.7, 0, 0.0, 0.0]
#         goal_pose = [0, -0.39, 10, 0, 0, 0]                
#         table_pose_dim = [0.0, -0.6, 0, 0, 0, 0, 0.3, 0.3, 0.]
#         object_asset_options = gymapi.AssetOptions()
#         object_asset_options.density = 500
#         object_asset_options.fix_base_link = True
#         object_asset_options.disable_gravity = True
#         object_asset_options.use_mesh_materials = True
#         object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
#         object_asset_options.override_com = True
#         object_asset_options.override_inertia = True
#         object_asset_options.vhacd_enabled = True
#         object_asset_options.vhacd_params = gymapi.VhacdParams()
#         object_asset_options.vhacd_params.resolution = 200000
#         object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
#         object_type = "door"

#     if task_name == "door_close_inward":
#         hand_start_pose = [0.55, 0.2, 0.6, 3.14159, 1.57, 1.57]
#         another_hand_start_pose = [0.55, -0.2, 0.6, 3.14159, -1.57, 1.57]
#         object_pose = [0.0, 0., 0.7, 0, 3.14159, 0.0]
#         goal_pose = [0, -0.39, 10, 0, 0, 0]                
#         table_pose_dim = [0.0, -0.6, 0, 0, 0, 0, 0.3, 0.3, 0.]
#         object_asset_options = gymapi.AssetOptions()
#         object_asset_options.density = 500
#         object_asset_options.fix_base_link = True
#         object_asset_options.disable_gravity = True
#         object_asset_options.use_mesh_materials = True
#         object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
#         object_asset_options.override_com = True
#         object_asset_options.override_inertia = True
#         object_asset_options.vhacd_enabled = True
#         object_asset_options.vhacd_params = gymapi.VhacdParams()
#         object_asset_options.vhacd_params.resolution = 100000
#         object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
#         object_type = "door"

#     return hand_start_pose, another_hand_start_pose, object_pose, goal_pose, table_pose_dim, object_asset_options, object_type

# @torch.jit.script
# def compute_hand_reward(
#     rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
#     max_episode_length: float, object_pos, object_rot, target_pos, target_rot, object_left_handle_pos, object_right_handle_pos,
#     left_hand_pos, right_hand_pos, right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_lf_pos, right_hand_th_pos,
#     left_hand_ff_pos, left_hand_mf_pos, left_hand_rf_pos, left_hand_lf_pos, left_hand_th_pos,
#     dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
#     actions, action_penalty_scale: float,
#     success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
#     fall_penalty: float, max_consecutive_successes: int, av_factor: float, this_task: str
# ):
#     # Distance from the hand to the object
#     if this_task in ["catch_underarm", "hand_over", "catch_abreast", "catch_over2underarm"]:
#         goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)

#         # Orientation alignment for the cube in hand and goal cube
#         quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
#         rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

#         dist_rew = goal_dist
#         # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

#         action_penalty = torch.sum(actions ** 2, dim=-1)

#         # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
#         reward = torch.exp(-0.2*(dist_rew * dist_reward_scale + rot_dist))

#         # Find out which envs hit the goal and update successes count
#         goal_resets = torch.where(torch.abs(goal_dist) <= 0.03, torch.ones_like(reset_goal_buf), reset_goal_buf)
#         successes = successes + goal_resets

#         # Success bonus: orientation is within `success_tolerance` of goal orientation
#         reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

#         # Fall penalty: distance to the goal is larger than a threashold
#         reward = torch.where(object_pos[:, 2] <= 0.3, reward + fall_penalty, reward)

#         # Check env termination conditions, including maximum success number
#         resets = torch.where(object_pos[:, 2] <= 0.3, torch.ones_like(reset_buf), reset_buf)
#         if max_consecutive_successes > 0:
#             # Reset progress buffer on goal envs if max_consecutive_successes > 0
#             progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
#             resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
#         resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

#         # Apply penalty for not reaching the goal
#         if max_consecutive_successes > 0:
#             reward = torch.where(progress_buf >= max_episode_length, reward + 0.5 * fall_penalty, reward)

#         num_resets = torch.sum(resets)
#         finished_cons_successes = torch.sum(successes * resets.float())

#         cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

#         return reward, resets, goal_resets, progress_buf, successes, cons_successes

#     if this_task in ["door_open_outward"]:
#         # Distance from the hand to the object
#         goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
#         # goal_dist = target_pos[:, 2] - object_pos[:, 2]

#         right_hand_dist = torch.norm(object_right_handle_pos - right_hand_pos, p=2, dim=-1)
#         left_hand_dist = torch.norm(object_left_handle_pos - left_hand_pos, p=2, dim=-1)

#         right_hand_finger_dist = (torch.norm(object_right_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(object_right_handle_pos - right_hand_mf_pos, p=2, dim=-1)
#                                 + torch.norm(object_right_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(object_right_handle_pos - right_hand_lf_pos, p=2, dim=-1) 
#                                 + torch.norm(object_right_handle_pos - right_hand_th_pos, p=2, dim=-1))
#         left_hand_finger_dist = (torch.norm(object_left_handle_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(object_left_handle_pos - left_hand_mf_pos, p=2, dim=-1)
#                                 + torch.norm(object_left_handle_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(object_left_handle_pos - left_hand_lf_pos, p=2, dim=-1) 
#                                 + torch.norm(object_left_handle_pos - left_hand_th_pos, p=2, dim=-1))
        

#         # Orientation alignment for the cube in hand and goal cube
#         # quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
#         # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

#         right_hand_dist_rew = right_hand_finger_dist
#         left_hand_dist_rew = left_hand_finger_dist

#         # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

#         action_penalty = torch.sum(actions ** 2, dim=-1)

#         # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
#         # reward = torch.exp(-0.05*(up_rew * dist_reward_scale)) + torch.exp(-0.05*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.05*(left_hand_dist_rew * dist_reward_scale))
#         up_rew = torch.zeros_like(right_hand_dist_rew)
#         up_rew = torch.where(right_hand_finger_dist < 0.5,
#                         torch.where(left_hand_finger_dist < 0.5,
#                                         torch.abs(object_right_handle_pos[:, 1] - object_left_handle_pos[:, 1]) * 2, up_rew), up_rew)
#         # up_rew =  torch.where(right_hand_finger_dist <= 0.3, torch.norm(bottle_cap_up - bottle_pos, p=2, dim=-1) * 30, up_rew)

#         # reward = torch.exp(-0.1*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.1*(left_hand_dist_rew * dist_reward_scale))
#         reward = 2 - right_hand_dist_rew - left_hand_dist_rew + up_rew

#         resets = torch.where(right_hand_finger_dist >= 1.5, torch.ones_like(reset_buf), reset_buf)
#         resets = torch.where(left_hand_finger_dist >= 1.5, torch.ones_like(resets), resets)
#         # resets = torch.where(left_hand_dist >= 0.2, torch.ones_like(resets), resets)

#         # print(right_hand_dist_rew[0])
#         # print(left_hand_dist_rew[0])
#         # print(up_rew[0])

#         # Find out which envs hit the goal and update successes count
#         resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

#         goal_resets = torch.zeros_like(resets)

#         num_resets = torch.sum(resets)
#         finished_cons_successes = torch.sum(successes * resets.float())

#         cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

#         return reward, resets, goal_resets, progress_buf, successes, cons_successes

#     if this_task in ["lift_pot"]:
#         # Distance from the hand to the object
#         goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
#         # goal_dist = target_pos[:, 2] - object_pos[:, 2]
#         right_hand_dist = torch.norm(object_right_handle_pos - right_hand_pos, p=2, dim=-1)
#         left_hand_dist = torch.norm(object_right_handle_pos - left_hand_pos, p=2, dim=-1)
#         # Orientation alignment for the cube in hand and goal cube
#         # quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
#         # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

#         right_hand_dist_rew = right_hand_dist
#         left_hand_dist_rew = left_hand_dist

#         # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

#         action_penalty = torch.sum(actions ** 2, dim=-1)

#         # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
#         # reward = torch.exp(-0.05*(up_rew * dist_reward_scale)) + torch.exp(-0.05*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.05*(left_hand_dist_rew * dist_reward_scale))
#         up_rew = torch.zeros_like(right_hand_dist_rew)
#         up_rew = torch.where(right_hand_dist < 0.08,
#                             torch.where(left_hand_dist < 0.08,
#                                             3*(0.985 - goal_dist), up_rew), up_rew)
        
#         reward = 0.2 - right_hand_dist_rew - left_hand_dist_rew + up_rew

#         resets = torch.where(object_pos[:, 2] <= 0.3, torch.ones_like(reset_buf), reset_buf)
#         resets = torch.where(right_hand_dist >= 0.2, torch.ones_like(resets), resets)
#         resets = torch.where(left_hand_dist >= 0.2, torch.ones_like(resets), resets)

#         # Find out which envs hit the goal and update successes count
#         resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

#         goal_resets = torch.zeros_like(resets)

#         num_resets = torch.sum(resets)
#         finished_cons_successes = torch.sum(successes * resets.float())

#         cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

#         return reward, resets, goal_resets, progress_buf, successes, cons_successes
