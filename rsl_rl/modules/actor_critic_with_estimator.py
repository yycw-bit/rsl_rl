# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.utils import resolve_nn_activation, unpad_trajectories


class ActorCriticRecurrentAndEstimator(ActorCritic):
    is_recurrent = True

    def __init__(
        self,
        num_base_obs,
        num_estimator_obs,
        num_privillege_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        estimator_dims = 256,
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_size=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )
        self.num_base_obs = num_base_obs
        self.num_estimator_obs = num_estimator_obs
        super().__init__(
            num_actor_obs=rnn_hidden_size,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )
        activation = resolve_nn_activation(activation)

        estimator_layers = []
        estimator_layers.append(nn.Linear(num_estimator_obs, estimator_dims[0]))
        estimator_layers.append(activation)
        for l in range(len(estimator_dims) - 1):
            estimator_layers.append(nn.Linear(estimator_dims[l], estimator_dims[l + 1]))
            estimator_layers.append(activation)
        estimator_layers.append(nn.Linear(estimator_dims[-1], num_privillege_obs))
        self.estimator = nn.Sequential(*estimator_layers)

        self.memory_a = Memory(num_base_obs+num_privillege_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic MLP")

    def reset(self, dones=None):
        self.memory_a.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        basic_input = observations[..., :self.num_base_obs]
        estimator_output = self.infer_priv_latent(observations[..., -self.num_estimator_obs:]).detach()
        backbone_input = torch.cat([basic_input, estimator_output], dim=-1)
        input_a = self.memory_a(backbone_input, masks, hidden_states)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations,**kwargs):
        basic_input = observations[..., :self.num_base_obs]
        estimator_output = self.infer_priv_latent(observations[..., -self.num_estimator_obs:]).detach()
        backbone_input = torch.cat([basic_input, estimator_output], dim=-1)
        input_a = self.memory_a(backbone_input, **kwargs)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        return super().evaluate(critic_observations)

    def get_hidden_states(self):
        return self.memory_a.hidden_states, None

    def infer_priv_latent(self, estimator_input):
        return self.estimator(estimator_input)


class Memory(torch.nn.Module):
    def __init__(self, input_size, type="lstm", num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0.0
