# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import warnings

from rsl_rl.modules import ActorCriticRMA
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorageRMA
from rsl_rl.utils import string_to_callable


class PPORMA:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    actor_critic: ActorCriticRMA
    """The actor critic module."""

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        dagger_update_freq=20,
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        # self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.ac_optimizer = optim.Adam(list(self.actor_critic.actor.parameters()) + list(self.actor_critic.critic.parameters()), lr=learning_rate)
        self.transition = RolloutStorageRMA.Transition()

        # # Adaptation
        # if self.actor_critic.actor.history_encoder != None:
        #     self.hist_encoder_optimizer = optim.Adam(self.actor_critic.actor.history_encoder.parameters(), lr=learning_rate)

        # Depth
        if self.actor_critic.depth_encoder != None and self.actor_critic.depth_actor != None:
            self.depth_encoder_optimizer = optim.Adam(self.actor_critic.depth_encoder.parameters(), lr=learning_rate)
            self.depth_actor_optimizer = optim.Adam(self.actor_critic.depth_actor.parameters(), lr=learning_rate)
        else:
            self.depth_encoder_optimizer = None
            self.depth_actor_optimizer = None

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        #
        self.counter = 0

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, scan_obs_shape, action_shape):
        # create rollout storage
        self.storage = RolloutStorageRMA(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, scan_obs_shape, action_shape, None, self.device,)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, scan_obs, hist_encoding=False):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs, scan_obs,  hist_encoding).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs, scan_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.transition.scan_observations = scan_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs, last_private_obs):
        # compute value for the last step
        last_values = self.actor_critic.evaluate(last_critic_obs, last_private_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch)

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # generator for mini batches
        if self.actor_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (
            obs_batch,
            critic_obs_batch,
            scan_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
        ) in generator:

            # number of augmentations per sample
            num_aug = 1
            # original batch size
            original_batch_size = obs_batch.shape[0]
            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the actor_critic with the new parameters
            # -- actor
            self.actor_critic.act(obs_batch, scan_obs_batch, hist_encoding=False, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, scan_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )

            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.actor_critic.action_mean[:original_batch_size]
            sigma_batch = self.actor_critic.action_std[:original_batch_size]
            entropy_batch = self.actor_critic.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.ac_optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Gradient step
            # -- For PPO
            self.ac_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(self.actor_critic.actor.parameters()) + list(self.actor_critic.critic.parameters()), self.max_grad_norm)
            self.ac_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        self.counter += 1
        # For explore
        # self.enforce_min_std()

        # -- Clear the storage
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_entropy

    def update_dagger(self):
        mean_hist_latent_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            private_obs_batch,
            history_obs_bath,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
        ) in generator:
            with torch.inference_mode():
                self.actor_critic.act(obs_batch, private_obs_batch, history_obs_bath, hist_encoding=True, masks=masks_batch, hidden_states=hid_states_batch[0])

            # Adaptation module update
            with torch.inference_mode():
                priv_latent_batch = self.actor_critic.actor.infer_priv_latent(private_obs_batch)
            hist_latent_batch = self.actor_critic.actor.infer_hist_latent(history_obs_bath)
            hist_latent_loss = (priv_latent_batch.detach() - hist_latent_batch).norm(p=2, dim=1).mean()
            self.hist_encoder_optimizer.zero_grad()
            hist_latent_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.actor.history_encoder.parameters(), self.max_grad_norm)
            self.hist_encoder_optimizer.step()

            mean_hist_latent_loss += hist_latent_loss.item()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_hist_latent_loss /= num_updates
        self.storage.clear()
        self.counter += 1
        return mean_hist_latent_loss

    def update_depth_encoder(self, depth_latent_batch, scandots_latent_batch):
        # Depth encoder ditillation
        depth_encoder_loss = (scandots_latent_batch.detach() - depth_latent_batch).norm(p=2, dim=1).mean()
        loss = depth_actor_loss
        self.depth_encoder_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.depth_encoder.parameters(), self.max_grad_norm)
        self.depth_encoder_optimizer.step()
        return depth_encoder_loss.item()

    def update_depth_actor(self, actions_student_batch, actions_teacher_batch):
        depth_actor_loss = (actions_teacher_batch.detach() - actions_student_batch).norm(p=2, dim=1).mean()
        loss = depth_actor_loss
        self.depth_actor_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.depth_actor.parameters(), self.max_grad_norm)
        self.depth_actor_optimizer.step()
        return depth_actor_loss.item()

    def update_depth_both(self, depth_latent_batch, scandots_latent_batch, actions_student_batch, actions_teacher_batch):
        depth_encoder_loss = (scandots_latent_batch.detach() - depth_latent_batch).norm(p=2, dim=1).mean()
        depth_actor_loss = (actions_teacher_batch.detach() - actions_student_batch).norm(p=2, dim=1).mean()
        loss = depth_encoder_loss + depth_actor_loss
        self.depth_encoder_optimizer.zero_grad()
        self.depth_actor_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_([*self.actor_critic.depth_actor.parameters(), *self.actor_critic.depth_encoder.parameters()], self.max_grad_norm)
        self.depth_actor_optimizer.step()
        self.depth_encoder_optimizer.step()
        return depth_encoder_loss.item(), depth_actor_loss.item()

    def enforce_min_std(self):
        current_std = self.actor_critic.std.detach()
        new_std = torch.max(current_std, self.min_policy_std).detach()
        self.actor_critic.std.data = new_std
