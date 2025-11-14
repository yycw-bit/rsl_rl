# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation

# History Encoder
class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        # self.device = device
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps
        self.n_dim = input_size // tsteps

        channel_size = 10
        # last_activation = nn.ELU()

        self.encoder = nn.Sequential( nn.Linear(self.n_dim , 3 * channel_size), self.activation_fn )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4), self.activation_fn,
                    nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn,
                    nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn, nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Flatten())
        else:
            raise(ValueError("tsteps must be 10, 20 or 50"))

        self.linear_output = nn.Sequential( nn.Linear(channel_size * 3, output_size), self.activation_fn )

    def forward(self, obs):
        # nd * T * n_proprio
        nd = obs.shape[0]
        T = self.tsteps
        n_proprio = self.n_dim
        obs = obs.view(nd, T, n_proprio)
        # print("obs device", obs.device)
        # print("encoder device", next(self.encoder.parameters()).device)
        projection = self.encoder(obs.reshape([nd * T, -1])) # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output

class Actor(nn.Module):
    def __init__(self, actor_hidden_dims, activation, num_actions, num_priv, num_hist, num_prop, priv_encoder_dims, hist_step, out_tanh=False):
        super().__init__()

        # Priv Encoder
        if len(priv_encoder_dims) > 0:
            priv_encoder_layers = []
            priv_encoder_layers.append(nn.Linear(num_priv, priv_encoder_dims[0]))
            priv_encoder_layers.append(activation)
            for l in range(len(priv_encoder_dims) - 1):
                # if l == len(priv_encoder_dims) - 1:
                #     priv_encoder_layers.append(nn.Linear(priv_encoder_dims[l], num_actions))
                #     # priv_encoder_layers.append(nn.Tanh())
                # else:
                priv_encoder_layers.append(nn.Linear(priv_encoder_dims[l], priv_encoder_dims[l + 1]))
                priv_encoder_layers.append(activation)
            self.priv_encoder = nn.Sequential(*priv_encoder_layers)
            priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv

        self.num_priv = num_priv
        self.num_hist = num_hist
        self.num_prop = num_prop

        # Hist Encoder
        self.history_encoder = StateHistoryEncoder(activation, num_hist, hist_step, priv_encoder_output_dim)

        # Action Backbone
        if len(actor_hidden_dims) > 0:
            actor_layers = []
            actor_layers.append(nn.Linear(num_prop + priv_encoder_output_dim, actor_hidden_dims[0]))
            actor_layers.append(activation)
            for l in range(len(actor_hidden_dims)):
                if l == len(actor_hidden_dims) - 1:
                    actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                    # actor_layers.append(nn.Tanh())
                else:
                    actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                    actor_layers.append(activation)
            if out_tanh:
                actor_layers.append(nn.Tanh())
            self.actor_backbone = nn.Sequential(*actor_layers)
            actor_backbone_output_dim = actor_hidden_dims[-1]
        else:
            self.actor_backbone = nn.Identity()
            actor_backbone_output_dim = mlp_input_dim_a + priv_encoder_output_dim

    def forward(self, obs, private_observations, history_observations, hist_encoding=False):
        if hist_encoding:
            latent = self.infer_hist_latent(history_observations)
        else:
            latent = self.infer_priv_latent(private_observations)
        backbone_input = torch.cat([obs, latent], dim=1)
        return self.actor_backbone(backbone_input)

    def infer_priv_latent(self, priv):
        return self.priv_encoder(priv)

    def infer_hist_latent(self, hist):
        return self.history_encoder(hist)

class Critic(nn.Module):
    def __init__(self, mlp_input_dim_c, critic_hidden_dims, activation, num_priv, num_hist, num_prop):
        super().__init__()

        self.num_priv = num_priv
        self.num_hist = num_hist
        self.num_prop = num_prop

        # Value
        if len(critic_hidden_dims) > 0:
            critic_layers = []
            critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
            critic_layers.append(activation)
            for l in range(len(critic_hidden_dims)):
                if l == len(critic_hidden_dims) - 1:
                    critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
                else:
                    critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                    critic_layers.append(activation)
            self.critic_backbone = nn.Sequential(*critic_layers)
            critic_backbone_output_dim = critic_hidden_dims[-1]
        else:
            self.critic_backbone = nn.Identity()
            critic_backbone_output_dim = mlp_input_dim_c

    def forward(self, obs):
        return self.critic_backbone(obs)

class ActorCriticROA(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_prop_obs,
        num_critic_obs,
        num_private_obs,
        num_history_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        priv_encoder_dims=[64, 20],
        hist_step=1,
        activation="elu",
        init_noise_std=1.0,
        out_tanh=False,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        # mlp_input_dim_a = num_actorbackbone_obs
        mlp_input_dim_c = num_critic_obs + num_private_obs

        self.actor = Actor(actor_hidden_dims, activation, num_actions, num_private_obs, num_history_obs, num_prop_obs, priv_encoder_dims, hist_step, out_tanh)
        self.critic = Critic(mlp_input_dim_c, critic_hidden_dims, activation, num_private_obs, num_history_obs, num_prop_obs)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

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

    def update_distribution(self, observations, private_observations, history_observations, hist_encoding):
        mean = self.actor(observations, private_observations, history_observations, hist_encoding)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, private_observations, history_observations, hist_encoding, **kwargs):
        self.update_distribution(observations, private_observations, history_observations, hist_encoding)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, private_observations, history_observations, hist_encoding):
        actions_mean = self.actor(observations, private_observations, history_observations, hist_encoding)
        return actions_mean

    def evaluate(self, critic_observations, private_obs, **kwargs):
        value = self.critic(torch.cat([critic_observations, private_obs], dim=-1))
        return value
