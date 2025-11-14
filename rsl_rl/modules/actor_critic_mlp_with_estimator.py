# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class ActorCriticMlpAndEstimator(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_base_obs,
        num_estimator_obs,
        num_privillege_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        estimator_dims=256,
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print( "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()]))
        super().__init__()
        activation = resolve_nn_activation(activation)

        estimator_layers = []
        estimator_layers.append(nn.Linear(num_estimator_obs, estimator_dims[0]))
        estimator_layers.append(activation)
        for l in range(len(estimator_dims) - 1):
            estimator_layers.append(nn.Linear(estimator_dims[l], estimator_dims[l + 1]))
            estimator_layers.append(activation)
        estimator_layers.append(nn.Linear(estimator_dims[-1], num_privillege_obs))
        self.estimator = nn.Sequential(*estimator_layers)

        mlp_input_dim_a = num_base_obs + num_estimator_obs + num_privillege_obs
        mlp_input_dim_c = num_critic_obs
        self.num_base_obs = num_base_obs
        self.num_estimator_obs= num_estimator_obs

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
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

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Estimator MLP: {self.estimator}")

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

    def act(self, observations, **kwargs):
        estimator_output = self.infer_priv_latent(observations[..., -self.num_estimator_obs:]).detach()
        backbone_input = torch.cat([observations, estimator_output], dim=-1)
        self.update_distribution(backbone_input)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        estimator_output = self.infer_priv_latent(observations[..., -self.num_estimator_obs:]).detach()
        backbone_input = torch.cat([observations, estimator_output], dim=-1)
        actions_mean = self.actor(backbone_input)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def infer_priv_latent(self, estimator_input):
        return self.estimator(estimator_input)
