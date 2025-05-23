import os
import numpy as np
from collections import deque
from copy import deepcopy
import itertools
import time
import statistics
import torch
import torch.nn as nn
from torch.optim import Adam
from torch import Tensor
from gym.spaces import Space
from torch.utils.tensorboard import SummaryWriter

from algorithms.sarl.sac import ReplayBuffer

from algorithms.sarl.sac import MLPActorCritic


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class SAC:
    
    #TODO： now，obs == state ？
    def __init__(self,
                 vec_env,
                 actor_critic = MLPActorCritic,
                 ac_kwargs=dict(),
                 num_transitions_per_env=8,
                 num_learning_epochs=5,
                 num_mini_batches=100,
                 replay_size=100000,
                 gamma=0.99,
                 polyak=0.99,
                 learning_rate=1e-3,
                 max_grad_norm =0.5,
                 entropy_coef=0.2,
                 use_clipped_value_loss=True,
                 reward_scale=1,
                 batch_size=32,
                 device='cpu',
                 sampler='random',
                 log_dir='run',
                 is_testing=False,
                 print_log=True,
                 apply_reset=False,
                 asymmetric=False
                 ):
        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.state_space, Space):
            raise TypeError("vec_env.state_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")


        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space

        self.device = device
        self.asymmetric = asymmetric
        self.learning_rate = learning_rate

        # SAC components
        self.vec_env = vec_env
        self.actor_critic = actor_critic(vec_env.observation_space, vec_env.action_space, **ac_kwargs).to(self.device)
        print(self.actor_critic)
        self.actor_critic_targ = deepcopy(self.actor_critic)

        self.storage = ReplayBuffer(vec_env.num_envs, replay_size, batch_size, num_transitions_per_env, self.observation_space.shape,
                                     self.state_space.shape, self.action_space.shape, self.device, sampler)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.actor_critic_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.actor_critic.q1.parameters(), self.actor_critic.q2.parameters())

        self.pi_optimizer = Adam(self.actor_critic.pi.parameters(), lr=self.learning_rate)
        self.q_optimizer = Adam(self.q_params, lr=self.learning_rate)

        #SAC parameters

        self.num_transitions_per_env = num_transitions_per_env
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.polyak = polyak
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.warm_up = True

        # Log
        self.log_dir = log_dir
        self.print_log = print_log
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0
        self.is_testing = is_testing
        self.current_learning_iteration = 0

        self.apply_reset = apply_reset

    def test(self,path):
        self.actor_critic.load_state_dict(torch.load(path))
        self.actor_critic.eval()

    def load(self,path):
        self.actor_critic.load_state_dict(torch.load(path))
        self.current_learning_iteration = int(path.split("_")[-1].split(".")[0])
        self.actor_critic.train()

    def save(self,path):
        torch.save(self.actor_critic.state_dict(),path)

    def run(self,num_learning_iterations, log_interval = 1):
        """
        the main loop of training.
        :param num_learning_iterations: the maximum number of training steps
        :param log_interval: the frequency of saving model
        :return: None
        """
        current_obs = self.vec_env.reset()
        current_states = self.vec_env.get_state()
        if self.is_testing:
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
            # plt.ion()
            while True:
                with torch.no_grad():
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                    # Compute the action
                    actions = self.actor_critic.act(current_obs,deterministic =True)
                    # Step the vec_environment
                    next_obs, rews, dones, infos = self.vec_env.step(actions)
                    
                    # plot
                    current_accum_rew += rews[0].item()
                    accum_rew_list.append(current_accum_rew)
                    # plt.clf()
                    # plt.plot(accum_rew_list)
                    # plt.xlim(0, 100)
                    # plt.xlabel('step')
                    # plt.ylim(0,50)
                    # plt.ylabel('accumulated reward')
                    # plt.axhline(y=current_accum_rew, color='c', linestyle='-')
                    # if len(accum_rew_list) == 1:
                    #     pass
                    # plt.pause(0.001)
                    if dones[0]:
                        current_accum_rew = 0
                        accum_rew_list = []
                    
                    # if self.vec_env.task.object_attribute.num_task_with_random:
                    #     current_obs.copy_(torch.cat((next_obs, torch.tensor(object_attri.astype(np.float32)).to(self.device)), 1))
                    # else:
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
            # plt.ioff()  
                    
                    # current_obs.copy_(next_obs)
        else:
            rewbuffer = deque(maxlen=self.num_transitions_per_env)
            lenbuffer = deque(maxlen=self.num_transitions_per_env)
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)

            reward_sum = []
            episode_length = []

            for it in range(self.current_learning_iteration, num_learning_iterations):
                start = time.time()
                ep_infos = []

                # Rollout
                for _ in range(self.num_transitions_per_env):
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                        current_states = self.vec_env.get_state()
                    # Compute the action
                    actions = self.actor_critic.act(current_obs)
                    # Step the vec_environment
                    next_obs, rews, dones, infos = self.vec_env.step(actions)
                    rews *= self.reward_scale
                    next_states = self.vec_env.get_state()
                    # Record the transition
                    self.storage.add_transitions(current_obs, current_states, actions, rews,next_obs, dones)
                    current_obs.copy_(next_obs)
                    current_states.copy_(next_states)
                    # Book keeping
                    ep_infos.append(infos)

                    if self.print_log:
                        cur_reward_sum[:] += rews
                        cur_episode_length[:] += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                    if self.storage.step > self.batch_size:
                        self.warm_up = False

                    if self.warm_up == False:
                        mean_value_loss, mean_surrogate_loss = self.update()

                if self.print_log:
                    # reward_sum = [x[0] for x in reward_sum]
                    # episode_length = [x[0] for x in episode_length]
                    rewbuffer.extend(reward_sum)
                    lenbuffer.extend(episode_length)


                stop = time.time()
                collection_time = stop - start

                mean_trajectory_length, mean_reward = self.storage.get_statistics()

                # Learning step
                start = stop
                # TODO: need check the buffer size before update
                # add the update within the interaction loop
                if self.warm_up == False:
                    # mean_value_loss, mean_surrogate_loss = self.update()

                    stop = time.time()
                    learn_time = stop - start
                    if self.print_log:
                        self.log(locals())
                    if it % log_interval == 0:
                        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
                    ep_infos.clear()
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(num_learning_iterations)))

        pass

    def log(self, locs, width=80, pad=35):
        """
        print training info
        :param locs:
        :param width:
        :param pad:
        :return:
        """
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

        fps = int(self.num_transitions_per_env * self.vec_env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
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
        # for obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
        #        in self.storage.mini_batch_generator(self.num_mini_batches):
        #TODO: sample a random indice of the batch
        # as now the training uses the whole dataset
        for epoch in range(self.num_learning_epochs):
            # learn_ep = 0
            for indices in batch:
                # learn_ep += 1
                
                # if learn_ep >= self.num_learning_epochs:
                #     break

                obs_batch = self.storage.observations[indices]
                nextobs_batch = self.storage.next_observations[indices]
                if self.asymmetric:
                    states_batch = self.storage.states[indices]
                else:
                    states_batch = None
                actions_batch = self.storage.actions[indices]
                rewards_batch = self.storage.rewards[indices]
                dones_batch = self.storage.dones[indices]

                data = {'obs': obs_batch,
                        'act':actions_batch,
                        'r':rewards_batch,
                        'obs2':nextobs_batch,
                        'done':dones_batch}

                self.q_optimizer.zero_grad()
                loss_q = self.compute_loss_q(data)
                loss_q.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.q_optimizer.step()

                # Record things
                mean_value_loss += loss_q.item()

                # Freeze Q-networks so you don't waste computational effort
                # computing gradients for them during the policy learning step.
                for p in self.q_params:
                    p.requires_grad = False

                # Next run one gradient descent step for pi.
                self.pi_optimizer.zero_grad()
                loss_pi = self.compute_loss_pi(data)
                loss_pi.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.pi_optimizer.step()

                # Unfreeze Q-networks so you can optimize it at next DDPG step.
                for p in self.q_params:
                    p.requires_grad = True

                # Record things
                mean_surrogate_loss += loss_pi.item()

                # Finally, update target networks by polyak averaging.
                with torch.no_grad():
                    for p, p_targ in zip(self.actor_critic.parameters(), self.actor_critic_targ.parameters()):
                        # NB: We use an in-place operations "mul_", "add_" to update target
                        # params, as opposed to "mul" and "add", which would make new tensors.
                        p_targ.data.mul_(self.polyak)
                        p_targ.data.add_((1 - self.polyak) * p.data)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss

    def compute_loss_q(self,data):
        o, a, r, o2, d = data['obs'],data['act'], data['r'], data['obs2'], data['done']

        q1 = self.actor_critic.q1(o, a)
        q2 = self.actor_critic.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.actor_critic.pi(o2)

            # Target Q-values
            q1_pi_targ = self.actor_critic_targ.q1(o2, a2)
            q2_pi_targ = self.actor_critic_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = (r + self.gamma * (1 - d) * (q_pi_targ - self.entropy_coef * logp_a2))

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self,data):
        o = data['obs']
        pi, logp_pi = self.actor_critic.pi(o)

        q1_pi = self.actor_critic.q1(o, pi)
        q2_pi = self.actor_critic.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.entropy_coef * logp_pi - q_pi).mean()

        return loss_pi



