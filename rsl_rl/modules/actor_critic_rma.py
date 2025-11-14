# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import code
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from torch.nn.modules.activation import ReLU
from copy import copy, deepcopy
from rsl_rl.utils import resolve_nn_activation
from .depth_image_encoder import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone


class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        # self.device = device
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 10
        # last_activation = nn.ELU()

        self.encoder = nn.Sequential(
                nn.Linear(input_size, 3 * channel_size), self.activation_fn,
                )

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

        self.linear_output = nn.Sequential(
                nn.Linear(channel_size * 3, output_size), self.activation_fn
                )

    def forward(self, obs):
        # nd * T * n_proprio
        nd = obs.shape[0]
        T = self.tsteps
        # print("obs device", obs.device)
        # print("encoder device", next(self.encoder.parameters()).device)
        projection = self.encoder(obs.reshape([nd * T, -1])) # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output

class Actor(nn.Module):
    def __init__(self, actor_hidden_dims, activation, num_actions, scan_encoder_dims, num_hist, num_prop, num_scan, hist_step, out_tanh=False):

        super().__init__()
        # prop -> scan -> priv_explicit -> priv_latent -> hist
        # actor input: prop -> scan -> priv_explicit -> latent
        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.if_scan_encode = scan_encoder_dims is not None and num_scan > 0

        if len(scan_encoder_dims) > 0:
            scan_encoder = []
            scan_encoder.append(nn.Linear(num_scan, scan_encoder_dims[0]))
            scan_encoder.append(activation)
            for l in range(len(scan_encoder_dims) - 1):
                if l == len(scan_encoder_dims) - 2:
                    scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l + 1]))
                    scan_encoder.append(nn.Tanh())
                else:
                    scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l + 1]))
                    scan_encoder.append(activation)
            self.scan_encoder = nn.Sequential(*scan_encoder)
            self.scan_encoder_output_dim = scan_encoder_dims[-1]
        else:
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = num_scan

        if num_hist != None:
            self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, priv_encoder_output_dim)
        
        actor_layers = []
        actor_layers.append(nn.Linear(num_prop + self.scan_encoder_output_dim, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        if out_tanh:
            actor_layers.append(nn.Tanh())
        self.actor_backbone = nn.Sequential(*actor_layers)

    def forward(self, obs, scan_obs, hist_encoding=None, scandots_latent=None):
        if scandots_latent is None:
            scan_latent = self.infer_scandots_latent(scan_obs)
        else:
            scan_latent = scandots_latent
        obs_prop_scan = torch.cat([obs, scan_latent], dim=1)

        # if hist_encoding:
        #     latent = self.infer_hist_latent(obs)
        # else:
        #     latent = self.infer_priv_latent(obs)
        # backbone_input = torch.cat([obs, obs_priv_explicit, latent], dim=1)
        backbone_output = self.actor_backbone(obs_prop_scan)
        return backbone_output

    # def infer_priv_latent(self, obs):
    #     priv = obs[:, self.num_prop + self.num_scan + self.num_priv_explicit: self.num_prop + self.num_scan + self.num_priv_explicit + self.num_priv_latent]
    #     return self.priv_encoder(priv)
    
    # def infer_hist_latent(self, obs):
    #     hist = obs[:, -self.num_hist*self.num_prop:]
    #     return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))
    
    def infer_scandots_latent(self, obs):
        return self.scan_encoder(obs)

class Critic(nn.Module):
    def __init__(self, mlp_input_dim_c, critic_hidden_dims, activation, num_priv=None, num_hist=None, num_prop=None):
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

    def forward(self, obs, scan_obs):
        input = torch.cat([obs, scan_obs], dim=1)
        return self.critic_backbone(input)

class ActorCriticRMA(nn.Module):
    is_recurrent = False
    def __init__(self,  num_prop_obs,
                        num_critic_obs,
                        num_scan_obs,
                        num_history_obs,
                        num_actions,
                        depth_vis_boolean,
                        depth_encoder_dims=[256, 256, 256],
                        actor_hidden_dims=[256, 256, 256],
                        scan_encoder_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        out_tanh=False,
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticRMA, self).__init__()
        self.depth_vis_boolean = depth_vis_boolean
        self.kwargs = kwargs
        activation = resolve_nn_activation(activation)
        mlp_input_dim_c = num_scan_obs + num_critic_obs

        # Privillege Actor Net
        self.actor = Actor(actor_hidden_dims, activation, num_actions, scan_encoder_dims, num_history_obs, num_prop_obs, num_scan_obs, 0, out_tanh)

        if depth_vis_boolean:
            depth_backbone = DepthOnlyFCBackbone58x87(scan_encoder_dims[-1], depth_encoder_dims)
            self.depth_encoder = RecurrentDepthBackbone(depth_backbone, num_prop_obs)
            self.depth_actor = deepcopy(self.actor)
        else:
            self.depth_encoder = None
            self.depth_actor = None

        # Value function
        self.critic = Critic(mlp_input_dim_c, critic_hidden_dims, activation)

        print(f"Actor MLP: {self.actor}")
        print(f"Depth Encoder MLP: {self.depth_encoder}")
        print(f"Depth Actor MLP: {self.depth_actor}")
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
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

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

    def update_distribution(self, observations, scan_obs_batch, hist_encoding):
        mean = self.actor(observations, scan_obs_batch, hist_encoding)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, scan_obs_batch, hist_encoding=False, **kwargs):
        self.update_distribution(observations, scan_obs_batch, hist_encoding)
        return self.distribution.sample()

    def depth_act(self, observations, vision_obs_batch, hist_encoding=False, scandots_latent=None, **kwargs):
        depth_laten = self.depth_encoder(vision_obs_batch, observations)
        return self.depth_actor(observations, None, hist_encoding, depth_laten)
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, scan_obs_batch, hist_encoding=False, scandots_latent=None, **kwargs):
        actions_mean = self.actor(observations, scan_obs_batch, hist_encoding, scandots_latent)
        return actions_mean

    def evaluate(self, critic_observations, scan_obs_batch, **kwargs):
        value = self.critic(critic_observations, scan_obs_batch)
        return value
    
    def reset_std(self, std, num_actions, device):
        new_std = std * torch.ones(num_actions, device=device)
        self.std.data = new_std.data
