# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
from typing import Tuple
from isaacgym import gymapi, gymtorch
from torch import Tensor
from tasks.hand_arm_base.base_task import BaseTask
from utils.torch_jit_utils import *
from tasks.hand_base.change_obj_attribute import Obj_attribute


class TwoHandArmsPoint2Point(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.cfg = cfg

        self.num_arm_dofs = 6
        self.num_finger_dofs = 4
        self.num_hand_fingertips = 4
        self.num_hand_dofs = self.num_finger_dofs * self.num_hand_fingertips
        self.num_hand_arm_dofs = self.num_hand_dofs + self.num_arm_dofs
        self.hand_arm_asset_file: str = self.cfg["env"]["asset"]["hand_arm_asset"]
        self.num_arms = self.cfg["env"]["numArms"]
        assert self.num_arms == 2, f"Only two arms supported, got {self.num_arms}"

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.clamp_abs_observations: float = self.cfg["env"]["clampAbsObservations"]
        self.num_hand_arm_actions = self.num_hand_arm_dofs * self.num_arms
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]

        self.distance_delta_rew_scale = self.cfg["env"]["distanceDeltaRewScale"]
        self.lifting_rew_scale = self.cfg["env"]["liftingRewScale"]
        self.lifting_bonus = self.cfg["env"]["liftingBonus"]
        self.lifting_bonus_threshold = self.cfg["env"]["liftingBonusThreshold"]
        self.keypoint_rew_scale = self.cfg["env"]["keypointRewScale"]
        self.catch_arm_close_goal_rew_scale = self.cfg["env"]["catchArmCloseGoalRewScale"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.outside_punish = self.cfg["env"]["outsidePunish"]

        self.force_torque_obs_scale = 10.0
        self.reset_position_noise_x = self.cfg["env"]["resetPositionNoiseX"]
        self.reset_position_noise_y = self.cfg["env"]["resetPositionNoiseY"]
        self.reset_position_noise_z = self.cfg["env"]["resetPositionNoiseZ"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.success_steps: int = self.cfg["env"]["successSteps"]

        self.lifting_bonus_threshold = 0.55  # 0.55
        self.success_steps: int = 30  # oringin 15, set 40 in order to improve catch quality
        self.catch_arm_extra_ofs: float = 0.20
        self.catch_arm_x_right_board: float = -0.70 # -0.90
        self.throw_arm_x_left_board: float = 0.90
        self.catch_arm_y_board: float = 1.0
        self.throw_arm_y_board: float = 1.0
        self.catch_arm_z_up_board: float = 1.25
        self.throw_arm_z_up_board: float = 1.25
        self.catch_arm_z_down_board: float = 0.20
        self.throw_arm_z_down_board: float = 0.25
        self.max_consecutive_successes = 1 # 50

        self.num_envs = self.cfg["env"]["numEnvs"]

        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "table": "urdf/table_narrow.urdf",
            "ball": "urdf/objects/ball.urdf",
            "washer": "urdf/ycb_pybullet/blue_moon/model.urdf",
            "mug": "urdf/ycb_pybullet/mug/model.urdf",
            "banana": "urdf/ycb_pybullet/plastic_banana/model.urdf",
            "poker": "urdf/ycb_pybullet/poker_1/model.urdf",
            "stapler": "urdf/ycb_pybullet/stapler_1/model.urdf",
            "suger": "urdf/ycb_pybullet/suger_3/model.urdf",
            "bowl": "urdf/ycb_pybullet/bowl/model.urdf",
            "pie": "urdf/ycb_pybullet/orion_pie/model.urdf",
            "apple": "urdf/ycb_pybullet/plastic_apple/model.urdf",
            "peach": "urdf/ycb_pybullet/plastic_peach/model.urdf",
            "pear": "urdf/ycb_pybullet/plastic_pear/model.urdf",
            "strawberry": "urdf/ycb_pybullet/plastic_strawberry/model.urdf",
            "pen": "urdf/ycb_pybullet/blue_marker/model.urdf",
            "scissors": "urdf/ycb_pybullet/scissors/model.urdf",
        }
        # self.object_types = ["banana", "mug", "apple", "stapler", "suger", "bowl", "pie", "peach", "pear", "pen", "washer", "poker", "scissors"]
        self.object_types = ["mug", "apple", "peach", "pear"]
        self.num_asset = len(self.object_types)
        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["openai", "full_no_vel", "full", "full_state"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [openai, full_no_vel, full, full_state]")

        print("Obs type:", self.obs_type)

        self.full_state_size = 294
        num_states = self.full_state_size

        self.num_obs_dict = {
            "full_state": self.full_state_size,
        }

        # self.hand_fingertips = ["fingertip_link1", "fingertip_link2", "fingertip_link3", "thumb_fingertip_link"]
        # self.hand_fingertips = ["pip_link1", "pip_link2", "pip_link3", "thumb_fingertip_link"]
        self.hand_fingertips = ["pip_link1", "pip_link2", "pip_link3", "thumb_pip_link"]
        # self.fingertip_offsets = np.array([[0.0, -0.01, 0.04], [0.0, -0.01, 0.04], [0.0, -0.01, 0.04], [0.06, 0.015, 0.0]], dtype=np.float32) # true
        self.fingertip_offsets = np.array([[0.0, -0.01, -0.04], [0.0, -0.01, -0.04], [0.0, -0.01, -0.04], [0.06, 0.015, 0.0]], dtype=np.float32)
        self.palm_offset = np.array([0.00, 0.12, 0.00], dtype=np.float32)
        assert self.num_hand_fingertips == len(self.hand_fingertips)

        self.up_axis = "z"
        self.use_vel_obs = False
        self.fingertip_obs = True

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        self.cfg["env"]["numActions"] = self.num_hand_arm_actions

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        super().__init__(cfg=self.cfg, is_meta=False)

        if self.viewer is not None:
            cam_pos = gymapi.Vec3(0.0, 1.8, 2.0)
            cam_target = gymapi.Vec3(-0.2, 0.0, 0.4)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.hand_arm_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, : self.num_hand_arm_dofs * self.num_arms]
        # this will have dimensions [num_envs, num_arms * num_hand_arm_dofs]
        self.hand_arm_dof_pos = self.hand_arm_dof_state[..., 0]
        self.hand_arm_dof_vel = self.hand_arm_dof_state[..., 1]
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.hand_arm_default_dof_pos = torch.zeros([self.num_arms, self.num_hand_arm_dofs], dtype=torch.float, device=self.device)
        desired_hand_pos_0 = torch.tensor([-1.571, -2.8, 2.2, -2.53, -1.47, 0.])
        desired_hand_pos_1 = torch.tensor([-1.571, -2.8, 2.2, -2.53, -1.47, 0.])
        self.hand_arm_default_dof_pos[0, :6] = desired_hand_pos_0      
        self.hand_arm_default_dof_pos[1, :6] = desired_hand_pos_1
        self.hand_arm_default_dof_pos = self.hand_arm_default_dof_pos.flatten()

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.palm_center_offset = torch.from_numpy(self.palm_offset).to(self.device).repeat((self.num_envs, 1))
        self.palm_center_pos = torch.zeros((self.num_envs, self.num_arms, 3), dtype=torch.float, device=self.device)
        self.fingertip_offsets = torch.from_numpy(self.fingertip_offsets).to(self.device).repeat((self.num_envs, 1, 1))

        self.prev_targets = torch.zeros((self.num_envs, self.num_arms * self.num_hand_arm_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_arms * self.num_hand_arm_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.prev_episode_successes = torch.zeros_like(self.successes)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.total_successes = 0
        self.total_resets = 0

        # core variables for rewards calculation
        self.near_goal_steps = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.lifted_object = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.closest_keypoint_max_dist = -torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        self.closest_catch_arm_goal_dist = -torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        self.last_fingertip_dist = -torch.ones([self.num_envs, self.num_arms, self.num_hand_fingertips], dtype=torch.float, device=self.device)
        self.last_fingertip_pos_offset = torch.zeros([self.num_envs, self.num_arms, self.num_hand_fingertips, 3], dtype=torch.float, device=self.device)
        self.closest_fingertip_dist = -torch.ones([self.num_envs, self.num_arms, self.num_hand_fingertips], dtype=torch.float, device=self.device)
        self.fingertip_pos_offset = torch.zeros([self.num_envs, self.num_arms * self.num_hand_fingertips, 3], dtype=torch.float, device=self.device)
        self.last_palm_dist = -torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        # help to differ the rewards of different arms
        self.throw_arm = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.catch_arm = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        reward_keys = [
            # "raw_fingertip_delta_rew",
            # "raw_lifting_rew",
            # "raw_keypoint_rew",
            "fingertip_delta_rew",
            "lifting_rew",
            "lift_bonus_rew",
            "keypoint_rew",
            "bonus_rew",
            "total_reward",
            "catch_palm_close_goal_rew",
            "outside_punish",
            "drop_penalty"
        ]
        self.rewards_episode = {key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for key in reward_keys}

        self.total_successes = 0
        self.total_resets = 0
    
    def create_sim(self):
        self.sim_params = self.parse_sim_params(self.sim_params, self.cfg["sim"])
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "../assets"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.vhacd_enabled = True

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        hand_arm_asset = self.gym.load_asset(self.sim, asset_root, self.hand_arm_asset_file, asset_options)

        num_hand_arm_bodies = self.gym.get_asset_rigid_body_count(hand_arm_asset)
        num_hand_arm_shapes = self.gym.get_asset_rigid_shape_count(hand_arm_asset)
        num_hand_arm_dofs = self.gym.get_asset_dof_count(hand_arm_asset)
        assert (self.num_hand_arm_dofs == num_hand_arm_dofs), f"Number of DOFs in asset {hand_arm_asset} is {num_hand_arm_dofs}, but {self.num_hand_arm_dofs} was expected"

        hand_arm_dof_props = self.gym.get_asset_dof_properties(hand_arm_asset)
        hand_arm_dof_lower_limits = []
        hand_arm_dof_upper_limits = []
        for arm_idx in range(self.num_arms):
            for i in range(self.num_hand_arm_dofs):
                hand_arm_dof_lower_limits.append(hand_arm_dof_props["lower"][i])
                hand_arm_dof_upper_limits.append(hand_arm_dof_props["upper"][i])
        self.hand_arm_dof_lower_limits = to_torch(hand_arm_dof_lower_limits, device=self.device)
        self.hand_arm_dof_upper_limits = to_torch(hand_arm_dof_upper_limits, device=self.device)

        # set arms
        arm_poses = [gymapi.Transform() for _ in range(self.num_arms)]
        arm_x_ofs = self.cfg["env"]["armXOfs"]
        arm_y_ofs = self.cfg["env"]["armYOfs"]
        for arm_idx, arm_pose in enumerate(arm_poses):
            x_ofs = arm_x_ofs * (-1 if arm_idx == 0 else 1) + (self.catch_arm_extra_ofs if arm_idx == 0 else 0)
            arm_pose.p = gymapi.Vec3(*get_axis_params(0.0, self.up_axis_idx)) + gymapi.Vec3(x_ofs, arm_y_ofs, 0)
            if arm_idx == 0:
                arm_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)
            else:
                arm_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), math.pi)

        # set objects
        self.object_attribute = Obj_attribute(num_envs=self.num_envs)
        object_assets = []
        object_asset_options = gymapi.AssetOptions()
        for i in range(self.num_asset):
            object_asset = self.gym.load_asset(self.sim, asset_root, self.asset_files_dict[self.object_types[i]], object_asset_options)
            object_assets.append(object_asset)
        object_rb_count = self.gym.get_asset_rigid_body_count(object_assets[0]) 
        object_shapes_count = self.gym.get_asset_rigid_shape_count(object_assets[0]) 

        # set table
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = False
        table_asset_options.fix_base_link = True
        table_asset = self.gym.load_asset(self.sim, asset_root, self.asset_files_dict["table"], table_asset_options)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3()
        table_pose.p.x = 0.3125
        table_pose_dy, table_pose_dz = 0.0, 0.38
        table_pose.p.y = arm_y_ofs + table_pose_dy
        table_pose.p.z = table_pose_dz
        table_rb_count = self.gym.get_asset_rigid_body_count(table_asset)
        table_shapes_count = self.gym.get_asset_rigid_shape_count(table_asset)

        # set goal objects
        goal_assets = []
        goal_asset_options = gymapi.AssetOptions()
        goal_asset_options.disable_gravity = True
        for i in range(self.num_asset):
            goal_asset = self.gym.load_asset(self.sim, asset_root, self.asset_files_dict[self.object_types[i]], goal_asset_options)
            goal_assets.append(goal_asset)
        goal_rb_count = self.gym.get_asset_rigid_body_count(goal_assets[0]) 
        goal_shapes_count = self.gym.get_asset_rigid_shape_count(goal_assets[0]) 

        # set up object and goal positions
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = 0.0
        pose_dy, pose_dz = table_pose_dy, table_pose_dz + 0.10
        object_start_pose.p.y = arm_y_ofs + pose_dy
        object_start_pose.p.z = pose_dz
        self.object_start_pose = object_start_pose

        self.envs = []
        object_init_state = []
        goal_object_indices = []

        hand_palm_handle = self.gym.find_asset_rigid_body_index(hand_arm_asset, "wrist3_Link")
        fingertip_handles = [self.gym.find_asset_rigid_body_index(hand_arm_asset, name) for name in self.hand_fingertips]

        self.hand_palm_handles = []
        self.hand_fingertip_handles = []
        self.extra_handle = []
        for arm_idx in range(self.num_arms):
            self.hand_palm_handles.append(hand_palm_handle + arm_idx * num_hand_arm_bodies)
            self.hand_fingertip_handles.extend([h + arm_idx * num_hand_arm_bodies for h in fingertip_handles])

        all_arms_bodies = num_hand_arm_bodies * self.num_arms
        self.object_rb_handles = list(range(all_arms_bodies, all_arms_bodies + object_rb_count))
        self.hand_arm_indices = torch.empty([self.num_envs, self.num_arms], dtype=torch.long, device=self.device)
        self.object_indices = torch.empty(self.num_envs, dtype=torch.long, device=self.device)

        max_agg_bodies = all_arms_bodies + object_rb_count + table_rb_count + goal_rb_count
        max_agg_shapes = num_hand_arm_shapes * self.num_arms + object_shapes_count + table_shapes_count + goal_shapes_count

        assert self.num_envs >= 1
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add arms
            for arm_idx in range(self.num_arms):
                hand_arm = self.gym.create_actor(env_ptr, hand_arm_asset, arm_poses[arm_idx], f"arm{arm_idx}", i, -1, 0)
                self.gym.set_actor_dof_properties(env_ptr, hand_arm, hand_arm_dof_props)
                hand_arm_idx = self.gym.get_actor_index(env_ptr, hand_arm, gymapi.DOMAIN_SIM)
                self.hand_arm_indices[i, arm_idx] = hand_arm_idx

            # add object
            object_asset_idx = (i + 0) % len(object_assets)
            object_asset = object_assets[object_asset_idx]

            obj_pose = self.object_start_pose
            object_handle = self.gym.create_actor(env_ptr, object_asset, obj_pose, "object", i, 0, 0)
            pos, rot = obj_pose.p, obj_pose.r
            object_init_state.append([pos.x, pos.y, pos.z, rot.x, rot.y, rot.z, rot.w, 0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices[i] = object_idx

            # table object
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table_object", i, 0, 0)
            _table_object_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)

            # goal object
            self.goal_displacement = gymapi.Vec3(-0.35, -0.06, 0.12)
            goal_start_pose = gymapi.Transform()
            goal_start_pose.p = self.object_start_pose.p + self.goal_displacement
            goal_start_pose.p.z -= 0.04
            goal_asset = goal_assets[object_asset_idx]
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            goal_object_indices.append(goal_object_idx)
            self.gym.set_actor_scale(env_ptr, goal_handle, 0.0001)
            
            self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)

        self.object_init_state = to_torch(object_init_state, device=self.device, dtype=torch.float).view(
            self.num_envs, 13
        )
        self.object_rebirth_state = self.object_init_state.clone()
        self.last_object_pos = self.object_init_state.clone()
        self.goal_states = self.object_init_state.clone()
        self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()

        self.hand_fingertip_handles = to_torch(self.hand_fingertip_handles, dtype=torch.long, device=self.device)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(goal_object_indices, dtype=torch.long, device=self.device)

    def _distance_delta_rewards(self, lifted_object: Tensor) -> Tensor:
        fingertip_dist_variation = self.last_fingertip_dist - self.curr_fingertip_distances
        # 当last_fingertip_dist为-1时，将variation设为0
        fingertip_dist_variation = torch.where(self.last_fingertip_dist == -1, torch.zeros_like(fingertip_dist_variation), fingertip_dist_variation)
        # print('curr_fingertip_distances', torch.mean(self.curr_fingertip_distances[0, 1, :]).item())
        throw_arm_fingertip_dist_variation = fingertip_dist_variation[torch.arange(self.num_envs), self.throw_arm, :]
        fingertip_deltas = torch.clip(throw_arm_fingertip_dist_variation, -5, 10)
        fingertip_delta_rew = torch.sum(fingertip_deltas, dim=-1)
        fingertip_delta_rew *= ~lifted_object
        self.last_fingertip_dist = self.curr_fingertip_distances.clone()

        return fingertip_delta_rew

    def _distance_delta_rewards_palm(self, lifted_object: Tensor) -> Tensor:
        # 计算掌心与物体的距离
        throw_arm_palm_pos = self.palm_center_pos[torch.arange(self.num_envs), self.throw_arm, :]
        palm_to_obj_dist = torch.norm(throw_arm_palm_pos - self.object_pos, dim=-1)
        
        # 计算距离变化
        palm_dist_variation = self.last_palm_dist - palm_to_obj_dist
        # 当last_palm_dist为-1时，将variation设为0
        palm_dist_variation = torch.where(self.last_palm_dist == -1, torch.zeros_like(palm_dist_variation), palm_dist_variation)
        palm_deltas = torch.clip(palm_dist_variation, -5, 10)
        palm_delta_rew = palm_deltas
        palm_delta_rew *= ~lifted_object
        
        # 更新上一时刻的距离
        self.last_palm_dist = palm_to_obj_dist.clone()

        return palm_delta_rew
    
    def _distance_delta_rewards_closest(self, lifted_object: Tensor) -> Tensor:
        fingertip_deltas_closest = self.closest_fingertip_dist - self.curr_fingertip_distances
        self.closest_fingertip_dist = torch.minimum(self.closest_fingertip_dist, self.curr_fingertip_distances)
        fingertip_deltas_closest = fingertip_deltas_closest[torch.arange(self.num_envs), self.throw_arm, :]
        fingertip_deltas = torch.clip(fingertip_deltas_closest, 0, 10)
        fingertip_delta_rew = torch.sum(fingertip_deltas, dim=-1)
        fingertip_delta_rew *= ~lifted_object

        return fingertip_delta_rew

    def _lifting_reward(self) -> Tuple[Tensor, Tensor, Tensor]:
        z_lift_all = -0.05 + self.object_pos[:, 2] - self.object_rebirth_state[:, 2]
        z_lift = self.object_pos[:, 2] - self.last_object_pos[:, 2]
        z_lift_used = torch.where(torch.mean(self.curr_fingertip_distances[torch.arange(self.num_envs), self.throw_arm, :], dim=-1) > 0.20, torch.zeros_like(z_lift), z_lift)
        # print('z_lift_used', z_lift_used[0].item())
        lifting_rew = torch.clip(z_lift_used, -0.5, 0.5)
        lifted_object = ((z_lift_all > self.lifting_bonus_threshold) &
                          (torch.mean(self.curr_fingertip_distances[torch.arange(self.num_envs), self.throw_arm, :], dim=-1) < 0.20)) | self.lifted_object
        just_lifted_above_threshold = lifted_object & ~self.lifted_object
        lift_bonus_rew = self.lifting_bonus * just_lifted_above_threshold
        lifting_rew *= ~lifted_object
        # lifting_rew *= (z_lift_all > 0.10)
        self.last_object_pos = self.object_pos.clone()
        self.lifted_object = lifted_object

        return lifting_rew, lift_bonus_rew, lifted_object
    
    def _catch_arm_close_goal_reward(self, lifted_object: Tensor) -> Tensor:
        catch_arm_keypoints_rel_palm = self.goal_kp_rel_palm[torch.arange(self.num_envs), self.catch_arm, :]
        catch_arm_keypoints_rel_palm_dist = torch.norm(catch_arm_keypoints_rel_palm, dim=-1)
        catch_arm_close_goal_delta = self.closest_catch_arm_goal_dist - catch_arm_keypoints_rel_palm_dist
        self.closest_catch_arm_goal_dist = torch.minimum(self.closest_catch_arm_goal_dist, catch_arm_keypoints_rel_palm_dist)
        catch_arm_kp_dist_rew = torch.clip(catch_arm_close_goal_delta, -10, 10)
        catch_arm_kp_dist_rew *= self.catch_arm_close_goal_rew_scale
        catch_arm_kp_dist_rew *= (lifted_object & (self.closest_catch_arm_goal_dist < 0.4))

        return catch_arm_kp_dist_rew

    def _keypoint_reward(self, lifted_object: Tensor) -> Tensor:
        max_keypoint_deltas = self.closest_keypoint_max_dist - self.keypoints_max_dist
        self.closest_keypoint_max_dist = torch.minimum(self.closest_keypoint_max_dist, self.keypoints_max_dist)
        max_keypoint_deltas = torch.clip(max_keypoint_deltas, 0, 100)
        keypoint_rew = max_keypoint_deltas * lifted_object

        return keypoint_rew

    def _compute_resets(self, is_success):
        resets = torch.where(self.object_pos[:, 2] < 0.1, torch.ones_like(self.reset_buf), self.reset_buf)
        if self.max_consecutive_successes > 0:
            self.progress_buf = torch.where(is_success > 0, torch.zeros_like(self.progress_buf), self.progress_buf)
            resets = torch.where(self.successes >= self.max_consecutive_successes, torch.ones_like(resets), resets)
        resets = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(resets), resets)
        catch_arm_palm_center_pos = self.palm_center_pos[torch.arange(self.num_envs), self.catch_arm, :]
        throw_arm_palm_center_pos = self.palm_center_pos[torch.arange(self.num_envs), self.throw_arm, :]
        catch_arm_palm_pos_reset_condition = (
            (catch_arm_palm_center_pos[:, 0] < self.catch_arm_x_right_board) |
            (catch_arm_palm_center_pos[:, 2] > self.catch_arm_z_up_board) |
            (catch_arm_palm_center_pos[:, 2] < self.catch_arm_z_down_board) |
            (catch_arm_palm_center_pos[:, 1] < -self.catch_arm_y_board) |
            (catch_arm_palm_center_pos[:, 1] > self.catch_arm_y_board)
        )
        
        throw_arm_palm_pos_reset_condition = (
            (throw_arm_palm_center_pos[:, 0] > self.throw_arm_x_left_board) |
            (throw_arm_palm_center_pos[:, 2] > self.throw_arm_z_up_board) |
            (throw_arm_palm_center_pos[:, 2] < self.throw_arm_z_down_board) |
            (throw_arm_palm_center_pos[:, 1] < -self.throw_arm_y_board) |
            (throw_arm_palm_center_pos[:, 1] > self.throw_arm_y_board)
        )
        condition = throw_arm_palm_pos_reset_condition | catch_arm_palm_pos_reset_condition
        resets = torch.where(condition, torch.ones_like(resets), resets)
        outside_punish = self.outside_punish * torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        outside_punish *= condition

        return resets, outside_punish

    def compute_point2point_reward(self) -> Tuple[Tensor, Tensor]:
        lifting_rew, lift_bonus_rew, lifted_object = self._lifting_reward()
        fingertip_delta_rew = self._distance_delta_rewards_palm(lifted_object) * 2.0
        catch_palm_close_goal_rew = self._catch_arm_close_goal_reward(lifted_object)
        keypoint_rew = self._keypoint_reward(lifted_object)

        near_goal: Tensor = self.keypoints_max_dist <= self.success_tolerance
        self.near_goal_steps += near_goal
        is_success = self.near_goal_steps >= self.success_steps
        goal_resets = is_success
        self.successes += is_success

        self.reset_goal_buf[:] = goal_resets

        # self.rewards_episode["raw_fingertip_delta_rew"] += fingertip_delta_rew
        # self.rewards_episode["raw_lifting_rew"] += lifting_rew
        # self.rewards_episode["raw_keypoint_rew"] += keypoint_rew

        fingertip_delta_rew *= self.distance_delta_rew_scale
        lifting_rew *= self.lifting_rew_scale
        keypoint_rew *= self.keypoint_rew_scale
        bonus_rew = near_goal * (self.reach_goal_bonus / self.success_steps)
        
        # 添加物体掉落的惩罚
        drop_penalty = torch.where(self.object_pos[:, 2] < 0.1, torch.tensor(0.0, device=self.device), torch.zeros(self.num_envs, device=self.device))
        
        reward = fingertip_delta_rew + lifting_rew + lift_bonus_rew + keypoint_rew + bonus_rew + catch_palm_close_goal_rew + drop_penalty  # + outside_punish

        resets, outside_punish = self._compute_resets(is_success)
        reward += outside_punish
        self.rew_buf[:] = reward
        self.reset_buf[:] = resets

        self.extras["successes"] = self.prev_episode_successes # .mean()
        rewards = [
            (fingertip_delta_rew, "fingertip_delta_rew"),
            (lifting_rew, "lifting_rew"),
            (lift_bonus_rew, "lift_bonus_rew"),
            (keypoint_rew, "keypoint_rew"),
            (bonus_rew, "bonus_rew"),
            (reward, "total_reward"),
            (catch_palm_close_goal_rew, "catch_palm_close_goal_rew"),
            (outside_punish, "outside_punish"),
            (drop_penalty, "drop_penalty")
        ]
        episode_cumulative = dict()
        for rew_value, rew_name in rewards:
            self.rewards_episode[rew_name] += rew_value
            episode_cumulative[rew_name] = rew_value
        self.extras["rewards_episode"] = self.rewards_episode
        # print('fingertip_delta_rew', self.rewards_episode['fingertip_delta_rew'][0].item())
        # print('lifting_rew', self.rewards_episode['lifting_rew'][0].item())
        # self.extras["episode_cumulative"] = episode_cumulative

    def compute_observations(self) -> Tuple[Tensor, int]:
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.object_state = self.root_state_tensor[self.object_indices, 0:13]
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        self._palm_state = self.rigid_body_states[:, self.hand_palm_handles]
        palm_pos = self._palm_state[..., 0:3]
        self._palm_rot = self._palm_state[..., 3:7]
        for arm_idx in range(self.num_arms):
            self.palm_center_pos[:, arm_idx] = palm_pos[:, arm_idx] + quat_rotate(self._palm_rot[:, arm_idx], self.palm_center_offset)

        self.fingertip_state = self.rigid_body_states[:, self.hand_fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.fingertip_state[:, :, 0:3]
        self.fingertip_rot = self.fingertip_state[:, :, 3:7]

        self.last_fingertip_pos_offset = self.fingertip_pos_offset.clone()
        self.fingertip_pos_offset = torch.zeros_like(self.fingertip_pos).to(self.device)
        for arm_idx in range(self.num_arms):
            for i in range(self.num_hand_fingertips):
                finger_idx = arm_idx * self.num_hand_fingertips + i
                self.fingertip_pos_offset[:, finger_idx] = self.fingertip_pos[:, finger_idx] + quat_rotate(self.fingertip_rot[:, finger_idx], self.fingertip_offsets[:, i])

        obj_pos_repeat = self.object_pos.unsqueeze(1).repeat(1, self.num_arms * self.num_hand_fingertips, 1)
        self.fingertip_pos_rel_object = self.fingertip_pos_offset - obj_pos_repeat
        self.curr_fingertip_distances = torch.norm(self.fingertip_pos_rel_object.view(self.num_envs, self.num_arms, self.num_hand_fingertips, -1), dim=-1)
        self.last_fingertip_dist = torch.where(self.last_fingertip_dist < 0.0, self.curr_fingertip_distances, self.last_fingertip_dist)
        self.closest_fingertip_dist = torch.where(self.closest_fingertip_dist < 0.0, self.curr_fingertip_distances, self.closest_fingertip_dist)

        palm_center_repeat = self.palm_center_pos.unsqueeze(2).repeat(1, 1, self.num_hand_fingertips, 1)
        self.fingertip_pos_rel_palm = self.fingertip_pos_offset - palm_center_repeat.view(self.num_envs, self.num_arms * self.num_hand_fingertips, 3)  

        self.keypoints_rel_goal = self.object_pos - self.goal_pos
        self.last_keypoints_rel_goal = self.last_object_pos[:, 0:3].view(self.num_envs, -1, 3) - self.goal_pos

        palm_center_repeat = self.palm_center_pos.unsqueeze(2).repeat(1, 1, 1, 1)
        obj_kp_pos_repeat = self.object_pos.unsqueeze(1).repeat(1, self.num_arms, 1, 1).view(self.num_envs, self.num_arms, 1, 3)
        goal_kp_pos_repeat = self.goal_pos.unsqueeze(1).repeat(1, self.num_arms, 1, 1).view(self.num_envs, self.num_arms, 1, 3)
        self.keypoints_rel_palm = obj_kp_pos_repeat - palm_center_repeat
        self.keypoints_rel_palm = self.keypoints_rel_palm.view(self.num_envs, self.num_arms, 3)
        self.goal_kp_rel_palm = goal_kp_pos_repeat - palm_center_repeat
        self.goal_kp_rel_palm = self.goal_kp_rel_palm.view(self.num_envs, self.num_arms, 3)
        self.goal_kp_rel_palm[:, :, 2] -= 0.05 

        self.keypoint_distances_l2 = torch.norm(self.keypoints_rel_goal, dim=-1)
        self.keypoints_max_dist = self.keypoint_distances_l2.max(dim=-1).values
        self.closest_keypoint_max_dist = torch.where(self.closest_keypoint_max_dist < 0.0, self.keypoints_max_dist, self.closest_keypoint_max_dist)
        self.closest_catch_arm_goal_dist = torch.where(self.closest_catch_arm_goal_dist < 0.0, torch.norm(self.goal_kp_rel_palm[torch.arange(self.num_envs), self.catch_arm, :], dim=-1), self.closest_catch_arm_goal_dist)

        if self.obs_type == "full_state":
            self.compute_full_state(self.obs_buf)
        else:
            raise ValueError("Unkown observations type!")

    def compute_full_state(self, buf: Tensor) -> Tuple[int, int]:
        """
        Index       Description
        0-43        hand-arms dof pos
        44-87       hand-arms dof vel
        88-113      palms state
        114-126     object state
        127-230     fingertips state
        231-274     actions
        275-280     obj rel palm
        281-283     goal rel palm(catch_arm)
        284-286     goal pos
        287-289     obj rel goal
        290-293     others    
        """
        num_dofs = self.num_hand_arm_dofs * self.num_arms
        ofs: int = 0
        buf[:, ofs : ofs + num_dofs] = unscale(
            self.hand_arm_dof_pos[:, :num_dofs],
            self.hand_arm_dof_lower_limits[:num_dofs],
            self.hand_arm_dof_upper_limits[:num_dofs],
        )
        ofs += num_dofs
        buf[:, ofs : ofs + num_dofs] = self.hand_arm_dof_vel[:, :num_dofs]
        ofs += num_dofs
        num_palm_coords = 3 * self.num_arms
        buf[:, ofs : ofs + num_palm_coords] = self.palm_center_pos.view(self.num_envs, num_palm_coords)
        ofs += num_palm_coords
        num_palm_rot_vel_angvel = 10 * self.num_arms
        buf[:, ofs : ofs + num_palm_rot_vel_angvel] = self._palm_state[..., 3:13].reshape(self.num_envs, num_palm_rot_vel_angvel)
        ofs += num_palm_rot_vel_angvel
        buf[:, ofs : ofs + 13] = self.object_state[:, 0:13]
        ofs += 13
        fingertip_state_size = 13 * self.num_hand_fingertips * self.num_arms
        buf[:, ofs : ofs + fingertip_state_size] = self.fingertip_state.reshape(self.num_envs, fingertip_state_size)
        ofs += fingertip_state_size
        action_size = 44
        buf[:, ofs : ofs + action_size] = self.actions[:, :action_size]
        ofs += action_size
        keypoint_rel_palm_size = 3 * self.num_arms
        buf[:, ofs : ofs + keypoint_rel_palm_size] = self.keypoints_rel_palm.reshape(self.num_envs, keypoint_rel_palm_size)
        ofs += keypoint_rel_palm_size
        buf[:, ofs : ofs + 3] = self.goal_kp_rel_palm.reshape(self.num_envs, keypoint_rel_palm_size)[:, 0:3]
        ofs += 3
        buf[:, ofs : ofs + 3] = self.goal_pos
        ofs += 3
        keypoint_rel_pos_size = 3
        buf[:, ofs : ofs + keypoint_rel_pos_size] = self.keypoints_rel_goal.reshape(self.num_envs, keypoint_rel_pos_size)
        ofs += keypoint_rel_pos_size
        buf[:, ofs : ofs + 1] = self.closest_keypoint_max_dist.unsqueeze(-1)
        ofs += 1
        buf[:, ofs : ofs + 1] = self.lifted_object.unsqueeze(-1)
        ofs += 1
        buf[:, ofs : ofs + 1] = torch.log(self.progress_buf / 10 + 1).unsqueeze(-1)
        ofs += 1
        buf[:, ofs : ofs + 1] = torch.log(self.successes + 1).unsqueeze(-1)
        ofs += 1
        # reward_obs_ofs = ofs
        # ofs += 1
        assert ofs == self.full_state_size

    def reset_target_pose(self, env_ids, apply_reset=False) -> None:
        # self.reset_object_pose(env_ids)
        x_pos = torch_rand_float(0.15, 0.45, (len(env_ids), 1), device=self.device) # 0.15(0.15) 0.35(0.45)
        x_pos_sign = torch.where(self.catch_arm[env_ids].unsqueeze(-1) == 0, -torch.ones_like(x_pos), torch.ones_like(x_pos))
        x_pos = x_pos * x_pos_sign
        y_pos = torch_rand_float(-0.30, 0.30, (len(env_ids), 1), device=self.device) # (-0.20)-0.30 (0.20)0.30
        z_pos = torch_rand_float(0.50, 0.70, (len(env_ids), 1), device=self.device) # 0.40(0.50) 0.55(0.70)
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:1] = x_pos
        self.root_state_tensor[self.goal_object_indices[env_ids], 1:2] = y_pos
        self.root_state_tensor[self.goal_object_indices[env_ids], 2:3] = z_pos + 0.05
        self.goal_states[env_ids, 0:1] = x_pos
        self.goal_states[env_ids, 1:2] = y_pos
        self.goal_states[env_ids, 2:3] = z_pos + 0.05
        self.lifted_object[env_ids] = False

        self.reset_goal_buf[env_ids] = 0
        self.near_goal_steps[env_ids] = 0
        self.closest_keypoint_max_dist[env_ids] = -1
        self.closest_catch_arm_goal_dist[env_ids] = -1
        self.closest_fingertip_dist[env_ids] = -1
        # if apply_reset:
        #     goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
        #     self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                                  gymtorch.unwrap_tensor(self.root_state_tensor),
        #                                                  gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))

    def reset(self, env_ids):
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)
            
        self.reset_target_pose(env_ids)
        obj_indices = self.object_indices[env_ids]
        obj_goal_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                   self.goal_object_indices[env_ids]]).to(torch.int32))

        # reset object
        obj_x_ofs = 0.45
        x_pos = obj_x_ofs
        rand_pos_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        self.root_state_tensor[obj_indices] = self.object_init_state[env_ids].clone()

        self.root_state_tensor[obj_indices, 0:1] = x_pos + self.reset_position_noise_x * rand_pos_floats[:, 0:1]
        self.root_state_tensor[obj_indices, 1:2] = (
            self.object_init_state[env_ids, 1:2] + self.reset_position_noise_y * rand_pos_floats[:, 1:2]
        )
        self.root_state_tensor[obj_indices, 2:3] = (
            self.object_init_state[env_ids, 2:3] + self.reset_position_noise_z * rand_pos_floats[:, 2:3]
        )

        self.object_rebirth_state[env_ids, 0:3] = self.root_state_tensor[obj_indices, 0:3]
        self.last_object_pos[env_ids, 0:3] = self.object_rebirth_state[env_ids, 0:3]
        self.last_object_pos[env_ids, 2] += 0.075

        new_object_rot = self.get_ensure_quat(env_ids)
        self.root_state_tensor[obj_indices, 3:7] = new_object_rot
        self.root_state_tensor[obj_indices, 7:13] = torch.zeros_like(self.root_state_tensor[obj_indices, 7:13])

        self.last_fingertip_dist[env_ids] = -1

        # flattened list of arm actors that we need to reset
        hand_arm_indices = self.hand_arm_indices[env_ids].to(torch.int32).flatten()

        # reset allegro hand
        delta_max = self.hand_arm_dof_upper_limits - self.hand_arm_default_dof_pos
        delta_min = self.hand_arm_dof_lower_limits - self.hand_arm_default_dof_pos

        rand_dof_floats = torch_rand_float(
            0.0, 1.0, (len(env_ids), self.num_arms * self.num_hand_arm_dofs), device=self.device
        )

        rand_delta = delta_min + (delta_max - delta_min) * rand_dof_floats

        allegro_pos = self.hand_arm_default_dof_pos + self.reset_dof_pos_noise * rand_delta

        self.hand_arm_dof_pos[env_ids, ...] = allegro_pos
        self.prev_targets[env_ids, ...] = allegro_pos
        self.cur_targets[env_ids, ...] = allegro_pos

        rand_vel_floats = torch_rand_float(
            -1.0, 1.0, (len(env_ids), self.num_hand_arm_dofs * self.num_arms), device=self.device
        )
        self.hand_arm_dof_vel[env_ids, :] = self.reset_dof_vel_noise * rand_vel_floats

        hand_arm_indices_gym = gymtorch.unwrap_tensor(hand_arm_indices)
        num_hand_arm_indices: int = len(hand_arm_indices)
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.prev_targets), hand_arm_indices_gym, num_hand_arm_indices
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), hand_arm_indices_gym, num_hand_arm_indices
        )

        # all_indices = torch.unique(torch.cat([obj_goal_indices, hand_arm_indices]).to(torch.int32))
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(obj_goal_indices), len(obj_goal_indices))
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.prev_episode_successes[env_ids] = self.successes[env_ids]
        self.successes[env_ids] = 0
        self.lifted_object[env_ids] = False
        self.closest_keypoint_max_dist[env_ids] = -1
        self.closest_catch_arm_goal_dist[env_ids] = -1
        self.last_fingertip_dist[env_ids] = -1
        self.last_palm_dist[env_ids] = -1
        self.near_goal_steps[env_ids] = 0
        self.closest_fingertip_dist[env_ids] = -1
        for key in self.rewards_episode.keys():
            self.rewards_episode[key][env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # # if only goals need reset, then call set API
        # if len(goal_env_ids) > 0 and len(env_ids) == 0:
        #     self.reset_target_pose(goal_env_ids, apply_reset=True)
        # # if goals need reset in addition to other envs, call set API in reset()
        # elif len(goal_env_ids) > 0:
        #     self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset(env_ids)

        if self.use_relative_control:
            raise NotImplementedError("Use relative control False for now")
        else:
            # TODO: this uses simplified finger control compared to the original code of 1-hand env

            num_dofs: int = self.num_hand_arm_dofs * self.num_arms

            # target position control for the hand DOFs
            self.cur_targets[..., :num_dofs] = scale(
                actions[..., :num_dofs],
                self.hand_arm_dof_lower_limits[:num_dofs],
                self.hand_arm_dof_upper_limits[:num_dofs],
            )
            self.cur_targets[..., :num_dofs] = (
                self.act_moving_average * self.cur_targets[..., :num_dofs]
                + (1.0 - self.act_moving_average) * self.prev_targets[..., :num_dofs]
            )
            self.cur_targets[..., :num_dofs] = tensor_clamp(
                self.cur_targets[..., :num_dofs],
                self.hand_arm_dof_lower_limits[:num_dofs],
                self.hand_arm_dof_upper_limits[:num_dofs],
            )

        self.prev_targets[...] = self.cur_targets[...]

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_point2point_reward()

    def get_random_quat(self, env_ids):
        # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py
        # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L261

        uvw = torch_rand_float(0, 1.0, (len(env_ids), 3), device=self.device)
        q_w = torch.sqrt(1.0 - uvw[:, 0]) * (torch.sin(2 * np.pi * uvw[:, 1]))
        q_x = torch.sqrt(1.0 - uvw[:, 0]) * (torch.cos(2 * np.pi * uvw[:, 1]))
        q_y = torch.sqrt(uvw[:, 0]) * (torch.sin(2 * np.pi * uvw[:, 2]))
        q_z = torch.sqrt(uvw[:, 0]) * (torch.cos(2 * np.pi * uvw[:, 2]))
        new_rot = torch.cat((q_x.unsqueeze(-1), q_y.unsqueeze(-1), q_z.unsqueeze(-1), q_w.unsqueeze(-1)), dim=-1)

        return new_rot
    
    def get_ensure_quat(self, env_ids):
        quats = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(len(env_ids), 1)

        return quats