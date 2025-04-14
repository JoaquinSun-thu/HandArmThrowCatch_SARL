from datetime import datetime
import os
import time

from gym.spaces import Space
from matplotlib import pyplot as plt

import numpy as np
import statistics
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from algorithms.sarl.ppo_lyp import RolloutStorage
from algorithms.sarl.ppo_lyp import Lyapunov



class PPO:
# some trick:[policy entropy:set entropy_coef=0.01, orignal is 0.0; ]
    def __init__(self,
                 vec_env,
                 actor_critic_class,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 init_noise_std=1.0,
                 value_loss_coef=1.0,
                 entropy_coef=0.0, 
                 learning_rate=1e-3,
                 max_grad_norm=0.5,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=None,
                 model_cfg=None,
                 device='cpu',
                 sampler='sequential',
                 log_dir='run',
                 is_testing=False,
                 print_log=True,
                 apply_reset=False,
                 asymmetric=False,
                 ):

        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.state_space, Space):
            raise TypeError("vec_env.state_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")
        
        self.observation_space = vec_env.observation_space  # add 2-dim point cloud feature (add 1 mass and 6 inertia in model under this ) 
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space

        self.device = device
        self.asymmetric = asymmetric

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.step_size = learning_rate

        # PPO components
        self.vec_env = vec_env

        # no
        # self.vec_env.task.object_attribute.num_task_with_random = 0

        if self.vec_env.task.object_attribute.num_task_with_random:
            observation_space_shape = list(self.observation_space.shape)
            observation_space_shape[0] += 2
            observation_space_shape = tuple(observation_space_shape)
        else:
            observation_space_shape = self.observation_space.shape
        
        self.actor_critic = actor_critic_class(observation_space_shape, self.state_space.shape, self.action_space.shape,
                                               init_noise_std, model_cfg, asymmetric=asymmetric)
        self.actor_critic.to(self.device)
        self.lyp = Lyapunov(observation_space_shape, model_cfg)
        self.lyp.to(self.device)
        self.storage = RolloutStorage(self.vec_env.num_envs, num_transitions_per_env, observation_space_shape,
                                      self.state_space.shape, self.action_space.shape, self.device, sampler)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.lyp_optimizer = optim.Adam(self.lyp.parameters(), lr=learning_rate)

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.num_transitions_per_env = num_transitions_per_env
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Log
        self.log_dir = log_dir
        self.print_log = print_log
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0
        self.is_testing = is_testing
        self.current_learning_iteration = 0

        self.apply_reset = apply_reset

    def test(self, path):
        self.actor_critic.load_state_dict(torch.load(path)) #, map_location="cuda:0"
        self.actor_critic.eval()

    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path))
        self.current_learning_iteration = int(path.split("_")[-1].split(".")[0])
        self.actor_critic.train()

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def run(self, num_learning_iterations, log_interval=1):
        current_obs = self.vec_env.reset()
        current_states = self.vec_env.get_state()

        

        if self.vec_env.task.object_attribute.num_task_with_random:
            # for random mass and inertia 
            # Object_attri = np.concatenate((np.expand_dims(self.vec_env.task.object_attribute.masses, axis=1), 
            #     np.expand_dims(self.vec_env.task.object_attribute.ixx, axis=1),
            #     np.expand_dims(self.vec_env.task.object_attribute.ixy, axis=1), 
            #     np.expand_dims(self.vec_env.task.object_attribute.ixz, axis=1),
            #     np.expand_dims(self.vec_env.task.object_attribute.iyy, axis=1),
            #     np.expand_dims(self.vec_env.task.object_attribute.iyz, axis=1),
            #     np.expand_dims(self.vec_env.task.object_attribute.izz, axis=1),), axis=1)
            # object_attri = Object_attri
            # for i in range(self.vec_env.task.object_attribute.num_task_with_random - 1):
            #     object_attri = np.concatenate((object_attri, Object_attri), axis=0)
            # for  point feature
            if self.vec_env.task.object_attribute.num_task_with_random > 1 :
                label_idx =   np.where(np.squeeze(self.vec_env.task.object_attribute.point_label) == self.vec_env.task.object_attribute.obj_label[self.vec_env.task.object_type[0]]) 
                Object_attri = np.ones([self.vec_env.task.num_each_envs, 1]) * self.vec_env.task.object_attribute.point_feature[label_idx]
                object_attri = Object_attri
                for i in range(self.vec_env.task.object_attribute.num_task_with_random - 1):
                    label_idx =   np.where(np.squeeze(self.vec_env.task.object_attribute.point_label) == self.vec_env.task.object_attribute.obj_label[self.vec_env.task.object_type[i+1]]) 
                    Object_attri = np.ones([self.vec_env.task.num_each_envs, 1]) * self.vec_env.task.object_attribute.point_feature[label_idx]
                    object_attri = np.concatenate((object_attri, Object_attri), axis=0)
            else:
                # # for using in catch_overarm/catch_abreast/catch_underarm/meta_ml1_random
                # label_idx =   np.where(np.squeeze(self.vec_env.task.object_attribute.point_label) == self.vec_env.task.object_attribute.obj_label[self.vec_env.task.object_type]) 
                # Object_attri = np.ones([self.vec_env.task.num_envs, 1]) * self.vec_env.task.object_attribute.point_feature[label_idx]
                
                # for using in meta_ml1_random task but can support only one num_task
                label_idx =   np.where(np.squeeze(self.vec_env.task.object_attribute.point_label) == self.vec_env.task.object_attribute.obj_label[self.vec_env.task.object_type[0]]) 
                Object_attri = np.ones([self.vec_env.task.num_each_envs, 1]) * self.vec_env.task.object_attribute.point_feature[label_idx]
                
                object_attri = Object_attri
              
                   
            current_obs = torch.cat((current_obs, torch.tensor(object_attri.astype(np.float32)).to(self.device)), 1)

        
        

        if self.is_testing:
            # fail_test = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device).detach()
            # success_test = torch.ones(self.vec_env.num_envs, dtype=torch.float, device=self.device).detach()
            # test_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device).detach()
            # test_reward_mean = 0
            rewbuffer = deque(maxlen=10000)
            lenbuffer = deque(maxlen=10000)

            distbuffer = deque(maxlen=10000)
            

            # action_sum_list = []
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            # cur_action_torque_sum = []

            reward_sum = []
            episode_length = []

            success_dist = []   


            test_iterm = 0
            accum_rew_list = []
            current_accum_rew = 0
            plt.ion()
            while True:  
                with torch.no_grad():
                    if self.apply_reset:
                        # test_reward_mean = test_reward_sum.mean()
                        # test_mask = torch.gt(test_reward_sum, 18)
                        # test_result = torch.masked_select(test_reward_sum, test_mask)

                        
                        # print("############################################")
                        # print("test_mean_reward: %", test_reward_mean)
                        # print("test_success_rate: %.2f", test_result.size(dim=0)/self.vec_env.num_envs)

                        # test_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device).detach()
                        
                        current_obs = self.vec_env.reset()
                        if self.vec_env.task.object_attribute.num_task_with_random:
                            current_obs = torch.cat((current_obs, torch.tensor(object_attri.astype(np.float32)).to(self.device)), 1)
                    # Compute the action
                    actions = self.actor_critic.act_inference(current_obs)
                    # Step the vec_environment
                    next_obs, rews, dones, infos = self.vec_env.step(actions)
                    
                    # plot
                    current_accum_rew += rews[0].item()
                    accum_rew_list.append(current_accum_rew)
                    plt.clf()
                    plt.plot(accum_rew_list)
                    plt.xlim(0, 100)
                    plt.xlabel('step')
                    plt.ylim(0,50)
                    plt.ylabel('accumulated reward')
                    plt.axhline(y=current_accum_rew, color='c', linestyle='-')
                    if len(accum_rew_list) == 1:
                        pass
                    plt.pause(0.001)
                    if dones[0]:
                        current_accum_rew = 0
                        accum_rew_list = []
                    
                    if self.vec_env.task.object_attribute.num_task_with_random:
                        current_obs.copy_(torch.cat((next_obs, torch.tensor(object_attri.astype(np.float32)).to(self.device)), 1))
                    else:
                        current_obs.copy_(next_obs)

                    cur_reward_sum[:] += rews
                    # if len(action_sum_list) < 10000:
                    #     action_sum_list.extend(torch.sum(actions, dim=1).cpu().detach().numpy()) 
                    cur_episode_length[:] += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    # print(new_ids.size())
                    # print(cur_reward_sum.size())
                    reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().detach().numpy().tolist())
                    episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().detach().numpy().tolist())                        
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0
                    rewbuffer.extend(reward_sum)
                    lenbuffer.extend(episode_length)

                    # print(self.vec_env.task.object_pos.size())
                    success_obj_pos = np.squeeze(self.vec_env.task.object_pos[new_ids].cpu().detach().numpy(), axis=1)
                    # print(success_obj_pos.shape)
                    success_goal_pos = np.squeeze(self.vec_env.task.goal_pos[new_ids,:].cpu().detach().numpy(), axis=1)
                    # print(success_goal_pos.shape)
                    success_dist.extend(np.linalg.norm(success_obj_pos - success_goal_pos, ord=2, axis=1).tolist())
                    distbuffer.extend(success_dist)


                    test_distances = torch.tensor(distbuffer, device=self.device).detach()
                    # print(test_distances.size())
                    
                    
                    dist_succsess_num_005 = (test_distances < 0.05).nonzero(as_tuple=False).size(dim=0)
                    dist_succsess_num_010 = (test_distances < 0.10).nonzero(as_tuple=False).size(dim=0)
                    dist_succsess_num_015 = (test_distances < 0.15).nonzero(as_tuple=False).size(dim=0)

                    test_rewards = torch.tensor(rewbuffer, device=self.device).detach()
                    
                    success_num_15 = (test_rewards > 15).nonzero(as_tuple=False).size(dim=0)
                    success_num_20 = (test_rewards > 20).nonzero(as_tuple=False).size(dim=0)
                    success_num_25 = (test_rewards > 25).nonzero(as_tuple=False).size(dim=0)

                   

                    test_rewards_mean = test_rewards.mean()

                    # print(len(reward_sum))
                    # print(len(success_dist))

                    # self.writer.add_scalar('Rate/success_rate', dist_succsess_num_010/100., test_iterm)
                    # self.writer.add_scalar('Power/action_resultant_torque', )
                    test_iterm += 1
                    
                    if len(rewbuffer) > 0:
                        print("############################################")
                        print(f"test_mean_reward: {test_rewards_mean:.2f}")
                        print(f"test_success_rate(reward_15): {success_num_15/100.:.2f} %" )
                        print(f"test_success_rate(reward_20): {success_num_20/100.:.2f} %" )
                        print(f"test_success_rate(reward_25): {success_num_25/100.:.2f} %" )
                        # print(f"test_action_sum_num: {len(action_sum_list)}")
                        print(f"current_buffer_size(reward): {len(rewbuffer)}")

                        print(torch.max(test_distances))
                        print(torch.min(test_distances))
                        print(f"test_success_rate(distance_005): {dist_succsess_num_005/100.:.2f} %" )
                        print(f"test_success_rate(distance_010): {dist_succsess_num_010/100.:.2f} %" )
                        print(f"test_success_rate(distance_015): {dist_succsess_num_015/100.:.2f} %" )

                        print(f"current_buffer_size(distance): {len(distbuffer)}")
                        print(f"current_buffer_size: {len(rewbuffer)}")
            plt.ioff()   
        else:
            rewbuffer = deque(maxlen=100)
            lenbuffer = deque(maxlen=100)
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)

            reward_sum = []
            episode_length = []

            last_lyp_values = torch.zeros(self.vec_env.num_envs, 1, dtype=torch.float, device=self.device)
            self.lyp_values_standard_batches = torch.zeros(self.vec_env.num_envs * self.num_transitions_per_env // self.num_mini_batches, 1, dtype=torch.float, device=self.device)  # mini batch size

            for it in range(self.current_learning_iteration, num_learning_iterations):
                start = time.time()
                ep_infos = []
                

                # Rollout
                for _ in range(self.num_transitions_per_env):
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                        current_states = self.vec_env.get_state()
                        if self.vec_env.task.object_attribute.num_task_with_random:
                            current_obs = torch.cat((current_obs, torch.tensor(object_attri.astype(np.float32)).to(self.device)), 1)
                    # Compute the action
                    actions, actions_log_prob, values, mu, sigma = self.actor_critic.act(current_obs, current_states)
                    # Compute the lyapunov value
                    lyp_values = self.lyp.compute_lyp_value(current_obs).detach()
                    # Step the vec_environment
                    next_obs, rews, dones, infos = self.vec_env.step(actions)
                    next_states = self.vec_env.get_state()
                    # Record the transition
                    self.storage.add_transitions(current_obs, current_states, actions, rews, dones, values, actions_log_prob, mu, sigma, lyp_values)
                    if self.vec_env.task.object_attribute.num_task_with_random:
                        current_obs.copy_(torch.cat((next_obs, torch.tensor(object_attri.astype(np.float32)).to(self.device)), 1))
                    else:
                        current_obs.copy_(next_obs)
                    current_states.copy_(next_states)
                    # Book keeping
                    ep_infos.append(infos)

                    if self.print_log:
                        cur_reward_sum[:] += rews
                        cur_episode_length[:] += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        # reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        # episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().detach().numpy().tolist())
                        episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().detach().numpy().tolist())                        
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                if self.print_log:
                    # reward_sum = [x[0] for x in reward_sum]
                    # episode_length = [x[0] for x in episode_length]
                    rewbuffer.extend(reward_sum)
                    lenbuffer.extend(episode_length)

                _, _, last_values, _, _ = self.actor_critic.act(current_obs, current_states)
                stop = time.time()
                collection_time = stop - start

                mean_trajectory_length, mean_reward = self.storage.get_statistics()

                # Learning step
                start = stop
                mean_advantage, mean_intri_advantage, mean_lyp_advantage, mean_totle_advantage = self.storage.compute_returns(last_values, last_lyp_values, self.gamma, self.lam)
                ## update Lyapunov value function
                mean_lyp_value_loss = self.update_lyp(last_lyp_values)
                last_lyp_values.copy_(self.storage.lyp_values[-1])
                ## update Policy and Value function
                mean_value_loss, mean_surrogate_loss = self.update()
                self.storage.clear()
                stop = time.time()
                learn_time = stop - start
                if self.print_log:
                    self.log(locals())
                if it % log_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
                ep_infos.clear()
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(num_learning_iterations)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_transitions_per_env * self.vec_env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.actor_critic.log_std.exp().mean()

        fps = int(self.num_transitions_per_env * self.vec_env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/lyp_value_function', locs['mean_lyp_value_loss'], locs['it'])
        self.writer.add_scalar('Advantage/mean_advantage', locs['mean_advantage'], locs['it'])
        self.writer.add_scalar('Advantage/mean_intri_advantage', locs['mean_intri_advantage'], locs['it'])
        self.writer.add_scalar('Advantage/mean_lyp_advantage', locs['mean_lyp_advantage'], locs['it'])
        self.writer.add_scalar('Advantage/mean_totle_advantage', locs['mean_totle_advantage'], locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/FPS',fps,locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        self.writer.add_scalar('Train2/mean_reward/step', locs['mean_reward'], locs['it'])
        self.writer.add_scalar('Train2/mean_episode_length/episode', locs['mean_trajectory_length'], locs['it'])

        # fps = int(self.num_transitions_per_env * self.vec_env.num_envs / (locs['collection_time'] + locs['learn_time']))

        str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        for epoch in range(self.num_learning_epochs):
            # for obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
            #        in self.storage.mini_batch_generator(self.num_mini_batches):

            for indices in batch:
                obs_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                if self.asymmetric:
                    states_batch = self.storage.states.view(-1, *self.storage.states.size()[2:])[indices]
                else:
                    states_batch = None
                actions_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
                target_values_batch = self.storage.values.view(-1, 1)[indices]
                returns_batch = self.storage.returns.view(-1, 1)[indices]
                old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[indices]
                advantages_batch = self.storage.advantages.view(-1, 1)[indices]
                old_mu_batch = self.storage.mu.view(-1, self.storage.actions.size(-1))[indices]
                old_sigma_batch = self.storage.sigma.view(-1, self.storage.actions.size(-1))[indices]

                actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.actor_critic.evaluate(obs_batch,
                                                                                                                       states_batch,
                                                                                                                       actions_batch)

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':

                    kl = torch.sum(
                        sigma_batch - old_sigma_batch + (torch.square(old_sigma_batch.exp()) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch.exp())) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.step_size = max(1e-5, self.step_size / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.step_size = min(1e-2, self.step_size * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.step_size

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))

                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                   1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                    

                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                ### loss.backward()
                loss.backward(retain_graph=True) 
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss

    def update_lyp(self, last_lyp_values):
        mean_lyp_value_loss = 0

        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        
        for epoch in range(self.num_learning_epochs):
            # for obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
            #        in self.storage.mini_batch_generator(self.num_mini_batches):

            for indices in batch:
                obs_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                last_lyp_values_batch = self.storage.values.view(-1, 1)[indices]

                lyp_values_batch = self.lyp.compute_lyp_value(obs_batch)

                # lyapunov risk
                lyp_risk_loss = torch.mean(torch.max(-lyp_values_batch, self.lyp_values_standard_batches) + torch.max(self.lyp_values_standard_batches, 1/1 * (lyp_values_batch - last_lyp_values_batch)))

                # Gradient step
                self.lyp_optimizer.zero_grad()
                lyp_risk_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.lyp.parameters(), self.max_grad_norm)
                self.lyp_optimizer.step()

                mean_lyp_value_loss += lyp_risk_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_lyp_value_loss /= num_updates
        
        return mean_lyp_value_loss

                

        
       
            