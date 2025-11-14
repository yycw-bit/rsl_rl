# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class UnitreeGo2WRoughPPORunnerWithEstimatorCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = "unitree_go2w_rough_with_Estimator"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticRecurrentAndEstimator",
        init_noise_std=1.0,
        actor_hidden_dims=[128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu"
    )
    policy.estimator_dims=[256, 128]
    policy.rnn_hidden_size = 256
    policy.rnn_num_layers = 2
    policy.rnn_type = 'gru'
    policy.num_base_obs = 57
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    wandb_project = "Isaaclab-Go2W-RNN-Wall-RNN"
