# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class W1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = "w1_rough_vae"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    policy.out_tanh = False
    policy.class_name = "ActorCriticVAE"
    policy.hist_step = 5
    policy.num_latent = 16
    policy.encoder_hidden_dims = [128]
    policy.decoder_hidden_dims = [64, 128]
    policy.vae_sigma_min = 0.0
    policy.vae_sigma_max = 5.0
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
    algorithm.class_name = "PPOVAE"
    algorithm.vae_learning_rate = 1.0e-3
    algorithm.kl_weight= 1.0
    wandb_project = "W1-Rough-VAE-v0"

@configclass
class W1FlatPPORunnerCfg(W1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "w1_flat_vae"
        wandb_project = "W1-Flat-VAE-v0"
