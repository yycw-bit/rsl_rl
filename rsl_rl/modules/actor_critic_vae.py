# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from .vae_estimator import VAE

class ActorCriticVAE(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_prop_obs,
        num_critic_obs,
        num_privilege_obs,
        num_history_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        encoder_hidden_dims=[128],
        decoder_hidden_dims=[64, 20],
        hist_step=1,
        num_latent=0,
        activation_str="elu",
        init_noise_std=1.0,
        vae_sigma_min=0.0,
        vae_sigma_max=5.0,
        out_tanh=False,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation_str)

        mlp_input_dim_a = num_prop_obs + num_latent + 3
        mlp_input_dim_c = num_critic_obs + num_privilege_obs

        # Actor
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        if out_tanh:
            actor_layers.append(nn.Tanh())
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # VAE with constrained reparameterization
        self.vae = VAE(
            num_obs=num_prop_obs,
            num_history=hist_step,
            num_latent=num_latent,
            activation=activation_str,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            sigma_min=vae_sigma_min,
            sigma_max=vae_sigma_max
        )

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"VAE MLP", self.vae)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, history_observations, **kwargs):
        assert observations.min() >= -100.0 and observations.max() <= 100.0, "Input data should be normalized"
        assert history_observations.min() >= -100.0 and history_observations.max() <= 100.0, "Input data should be normalized"
        estimation, latent_params = self.vae(history_observations)
        z, v = estimation
        assert not torch.isnan(z).any(), "z contains NaN values"
        assert not torch.isinf(z).any(), "z contains Inf values"
        assert not torch.isnan(v).any(), "v contains NaN values"
        assert not torch.isinf(v).any(), "v contains Inf values"
        self.update_distribution(torch.cat((z, v, observations), dim=-1))
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, history_observations):
        estimation, latent_params = self.vae(history_observations)
        z, v = estimation
        actions_mean = self.actor(torch.cat((z, v, observations), dim=-1))
        return actions_mean

    def evaluate(self, critic_observations, privilege_observations, **kwargs):
        value = self.critic(torch.cat([critic_observations, privilege_observations], dim=-1))
        return value
