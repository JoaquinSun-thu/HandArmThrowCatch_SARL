import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler, SubsetRandomSampler


class RolloutStorage:

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, states_shape, actions_shape, device='cpu', sampler='sequential'):

        self.device = device
        self.sampler = sampler
        ## Add intrinsic stability
        self.alpha = 0.2 # (alpha + belta) is in [0,1]
        self.intrinsic_stable_standard = torch.zeros(num_envs, 1, device=self.device)
        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.states = torch.zeros(num_transitions_per_env, num_envs, *states_shape, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        # Lyaponov Value
        self.belta = 0.2
        self.lyp_values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.lyp_values_standard = torch.zeros(num_envs, 1, device=self.device)
        self.constrain_k = 1.
        self.constrain_alpha = 0.7
        self.lyp_values_constrain = torch.zeros_like(self.lyp_values_standard)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0

    def add_transitions(self, observations, states, actions, rewards, dones, values, actions_log_prob, mu, sigma, lyp_value):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        self.observations[self.step].copy_(observations)
        self.states[self.step].copy_(states)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.dones[self.step].copy_(dones.view(-1, 1))
        self.values[self.step].copy_(values)
        self.actions_log_prob[self.step].copy_(actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(mu)
        self.sigma[self.step].copy_(sigma)

        self.lyp_values[self.step].copy_(lyp_value)

        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, last_lyp_values, gamma, lam, current_iter, num_max_iter):
        advantage = 0
        intri_derivatives = 0
        intri_advantage = 0
        lyp_derivatives = 0
        lyp_advantage = 0


        mean_advantege = 0
        mean_intri_advantage = 0
        mean_lyp_advantage = 0
        mean_totle_advantage = 0

        self.constrain_k = 0.1 + (current_iter+1) / num_max_iter * 0.9
        self.lyp_values_constrain = - self.constrain_k * torch.pow(torch.abs(last_lyp_values), self.constrain_alpha)

        
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
                next_lyp_values = last_lyp_values
            else:
                next_values = self.values[step + 1]
                next_lyp_values = self.lyp_values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            ## Add intrinsic stability
            intri_derivatives = 1 * (next_values - self.values[step])
            intri_advantage = torch.min(self.intrinsic_stable_standard, intri_derivatives)
            ## Add Lyapunov stability
            lyp_derivatives = 1 * (next_lyp_values - self.lyp_values[step])
            lyp_advantage = torch.min(self.lyp_values_constrain, -lyp_derivatives)

            totle_advantage = self.belta * lyp_advantage + (1 - self.belta) * advantage

            self.returns[step] = totle_advantage + self.values[step]

            mean_advantege += ((1 - self.alpha - self.belta) * advantage.mean().item())
            mean_intri_advantage += (self.alpha * intri_advantage.mean().item())
            mean_lyp_advantage += (self.belta * lyp_advantage.mean().item())
            mean_totle_advantage += totle_advantage.mean().item()

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

        # Compute the mean of every part advantage
        mean_advantege /= self.num_transitions_per_env
        mean_intri_advantage /= self.num_transitions_per_env
        mean_lyp_advantage /= self.num_transitions_per_env
        mean_totle_advantage /= self.num_transitions_per_env

        return mean_advantege, mean_intri_advantage, mean_lyp_advantage, mean_totle_advantage

    def get_statistics(self):
        done = self.dones.cpu()
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        if self.sampler == "sequential":
            # For physics-based RL, each environment is already randomized. There is no value to doing random sampling
            # but a lot of CPU overhead during the PPO process. So, we can just switch to a sequential sampler instead
            subset = SequentialSampler(range(batch_size))
        elif self.sampler == "random":
            subset = SubsetRandomSampler(range(batch_size))

        batch = BatchSampler(subset, mini_batch_size, drop_last=True)
        return batch
