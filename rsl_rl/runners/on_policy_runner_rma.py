# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque

import rsl_rl
from rsl_rl.algorithms import PPORMA
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCriticRMA
from rsl_rl.utils import store_code_state


class OnPolicyRunnerRMA:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.depth_encoder_cfg = train_cfg["depth_encoder_cfg"]
        self.depth_vis_boolean = train_cfg["depth_vis_boolean"]
        self.device = device
        self.env = env

        # resolve dimensions of observations
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        if "critic" in extras["observations"]:
            num_critic_obs = extras["observations"]["critic"].shape[1]
        else:
            num_critic_obs = num_obs
        # if "history" in extras["observations"]:
        #     num_history_obs = extras["observations"]["history"].shape[1]
        # else:
        #     num_history_obs = 0
        if "scan" in extras["observations"]:
            num_scan_obs = extras["observations"]["scan"].shape[1]
        else:
            num_scan_obs = 0
        if "vision" in extras["observations"]:
            num_vision_obs = extras["observations"]["vision"].shape[1]
        else:
            num_vision_obs = 0

        actor_critic_class = eval(self.policy_cfg.pop("class_name"))  # ActorCritic
        actor_critic: ActorCriticRMA = actor_critic_class(
            num_obs, num_critic_obs, num_scan_obs, None, self.env.num_actions, self.depth_vis_boolean, self.depth_encoder_cfg, **self.policy_cfg
        ).to(self.device)

        # init algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))  # PPOROA
        self.alg: PPORMA = alg_class(actor_critic, device=self.device, **self.alg_cfg)

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
            # self.private_obs_normalizer = EmpiricalNormalization(shape=[num_private_obs], until=1.0e8).to(self.device)
            self.scan_obs_normalizer = EmpiricalNormalization(shape=[num_scan_obs], until=1.0e8).to(self.device)
            self.vision_obs_normalizer = EmpiricalNormalization(shape=[num_vision_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            # self.private_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.scan_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.vision_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_critic_obs],
            [num_scan_obs],
            [self.env.num_actions],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        # self.dagger_update_freq = self.alg_cfg["dagger_update_freq"]
        self.git_status_repos = [rsl_rl.__file__]

    def learn_with_scan(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # init metrics
        mean_value_loss = 0.
        mean_surrogate_loss = 0.
        mean_entropy = 0.

        # start learning
        obs, extras = self.env.get_observations()
        # critic_obs = extras["observations"].get("critic", obs)
        # obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        obs = self.obs_normalizer(obs)
        critic_obs = self.critic_obs_normalizer(extras["observations"]["critic"].to(self.device))
        # private_obs = self.private_obs_normalizer(extras["observations"]["private"].to(self.device))
        scan_obs = self.scan_obs_normalizer(extras["observations"]["scan"].to(self.device))
        self.scan_train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions from policy
                    actions = self.alg.act(obs, critic_obs, scan_obs)
                    # Step environment
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    # Move to the agent device
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    # Normalize observations
                    obs = self.obs_normalizer(obs)
                    # Extract critic observations and normalize
                    # if "critic" in infos["observations"]:
                    #     critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"].to(self.device))
                    # else:
                    #     critic_obs = obs
                    critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"].to(self.device))
                    scan_obs = self.scan_obs_normalizer(infos["observations"]["scan"].to(self.device))
                    # history_obs = self.history_obs_normalizer(infos["observations"]["history"].to(self.device))
                    # Process env step and store in buffer
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        # Update rewards
                        cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        # -- common
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs, scan_obs)

            # Update policy
            # Note: we keep arguments here since locals() loads them
            mean_value_loss, mean_surrogate_loss, mean_entropy = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Logging info and save checkpoint
            if self.log_dir is not None:
                # Log information
                self.log_scan(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()

            # Save code state
            if it == start_iter:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] :
            self.writer.save_model(log_dir, self.current_learning_iteration)


    def learn_with_vision(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # init metrics
        depth_encoder_loss = 0.
        depth_actor_loss = 0.

        # start learning
        obs, extras = self.env.get_observations()
        # critic_obs = extras["observations"].get("critic", obs)
        # obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        obs = self.obs_normalizer(obs)
        critic_obs = self.critic_obs_normalizer(extras["observations"]["critic"].to(self.device))
        # private_obs = self.private_obs_normalizer(extras["observations"]["private"].to(self.device))
        scan_obs = self.scan_obs_normalizer(extras["observations"]["scan"].to(self.device))
        vision_obs = self.scan_obs_normalizer(extras["observations"]["vision"].to(self.device))
        self.vision_train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            scandots_latent_buffer = []
            depth_latent_buffer = []
            actions_teacher_buffer = []
            actions_student_buffer = []
            # Rollout
            for _ in range(self.num_steps_per_env):
                depth_latent = self.alg.actor_critic.depth_encoder(vision_obs,
                                                                   obs)  ## clone is crucial to avoid in-place operation
                # obs_student = obs
                # Sample actions from policy
                actions_student = self.alg.actor_critic.depth_actor(obs, None, hist_encoding=False,
                                                                    scandots_latent=depth_latent)
                depth_latent_buffer.append(depth_latent)
                actions_student_buffer.append(actions_student)
                with torch.no_grad():
                    scandots_latent = self.alg.actor_critic.actor.infer_scandots_latent(scan_obs)
                    actions_teacher = self.alg.actor_critic.act_inference(obs, None, hist_encoding=False,
                                                                          scandots_latent=scandots_latent)
                scandots_latent_buffer.append(scandots_latent)
                actions_teacher_buffer.append(actions_teacher)

                # Step environment
                obs, rewards, dones, infos = self.env.step(actions_student.detach().to(self.env.device))
                # Move to the agent device
                obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                # Normalize observations
                obs = self.obs_normalizer(obs)
                # Extract critic observations and normalize
                # if "critic" in infos["observations"]:
                #     critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"].to(self.device))
                # else:
                #     critic_obs = obs
                critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"].to(self.device))
                scan_obs = self.scan_obs_normalizer(infos["observations"]["scan"].to(self.device))
                vision_obs = self.scan_obs_normalizer(extras["observations"]["vision"].to(self.device))
                # history_obs = self.history_obs_normalizer(infos["observations"]["history"].to(self.device))

                # Process env step and store in buffer
                # self.alg.process_env_step(rewards, dones, infos)

                if self.log_dir is not None:
                    # Book keeping
                    if "episode" in infos:
                        ep_infos.append(infos["episode"])
                    elif "log" in infos:
                        ep_infos.append(infos["log"])
                    # Update rewards
                    cur_reward_sum += rewards
                    # Update episode length
                    cur_episode_length += 1
                    # Clear data for completed episodes
                    # -- common
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    self.alg.actor_critic.depth_encoder.reset_hidden_states(new_ids.squeeze(-1))
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

            stop = time.time()
            collection_time = stop - start

            # Learning step
            start = stop
            # self.alg.compute_returns(critic_obs, scan_obs)

            # Update policy
            # Note: we keep arguments here since locals() loads them
            scandots_latent_buffer = torch.cat(scandots_latent_buffer, dim=0)
            depth_latent_buffer = torch.cat(depth_latent_buffer, dim=0)
            actions_teacher_buffer = torch.cat(actions_teacher_buffer, dim=0)
            actions_student_buffer = torch.cat(actions_student_buffer, dim=0)
            depth_encoder_loss, depth_actor_loss = self.alg.update_depth_both(depth_latent_buffer, scandots_latent_buffer,
                                                                              actions_student_buffer, actions_teacher_buffer)

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            self.alg.actor_critic.depth_encoder.detach_hidden_states()

            # Logging info and save checkpoint
            if self.log_dir is not None:
                # Log information
                self.log_vision(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()

            # Save code state
            if it == start_iter:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] :
            self.writer.save_model(log_dir, self.current_learning_iteration)


    def log_scan(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode info
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        # -- Losses
        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        # self.writer.add_scalar("Loss/hist_latent_loss", locs["mean_hist_latent_loss"], locs["it"])
        self.writer.add_scalar("Loss/entropy", locs["mean_entropy"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # -- Policy
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- Training
        if len(locs["rewbuffer"]) > 0:
            # everything else
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
            )

            log_string += f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            log_string += f"""{'Mean total reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
            )

            log_string += f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""

            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)


    def log_vision(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode info
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        # -- Losses
        self.writer.add_scalar("Loss/depth_encoder", locs["depth_encoder_loss"], locs["it"])
        self.writer.add_scalar("Loss/depth_actor", locs["depth_actor_loss"], locs["it"])

        # -- Policy
        # self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- Training
        if len(locs["rewbuffer"]) > 0:
            # everything else
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
            )

            log_string += f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            log_string += f"""{'Mean total reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            log_string += f"""{'Depth encoder loss:':>{pad}} {locs['depth_encoder_loss']:.4f}\n"""
            log_string += f"""{'Depth actor loss:':>{pad}} {locs['depth_actor_loss']:.4f}\n"""
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n""")

            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)


    def save(self, path: str, infos=None):
        # -- Save PPO model
        saved_dict = {
            "actor_state_dict": self.alg.actor_critic.actor.state_dict(),
            "critic_state_dict": self.alg.actor_critic.critic.state_dict(),
            "depth_encoder_state_dict": self.alg.actor_critic.depth_encoder.state_dict() if self.alg.actor_critic.depth_encoder!=None else None,
            "depth_actor_state_dict": self.alg.actor_critic.depth_actor.state_dict() if self.alg.actor_critic.depth_actor!=None else None,
            "ac_optimizer_state_dict": self.alg.ac_optimizer.state_dict(),
            "depth_encoder_optimizer_state_dict": self.alg.depth_encoder_optimizer.state_dict()if self.alg.depth_encoder_optimizer!=None else None,
            "depth_actor_optimizer_state_dict": self.alg.depth_actor_optimizer.state_dict()if self.alg.depth_actor_optimizer!=None else None,
            "iter": self.current_learning_iteration,
            "infos": infos
        }
        # -- Save observation normalizer if used
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()
        torch.save(saved_dict, path)

        # # Upload model to external logging service
        # if self.logger_type in ["neptune", "wandb"] and self.current_learning_iteration > 3999 and self.current_learning_iteration%1000 == 0:
        #     self.writer.save_model(path, self.current_learning_iteration)


    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        # -- Load PPO model
        self.alg.actor_critic.actor.load_state_dict(loaded_dict["actor_state_dict"])
        self.alg.actor_critic.critic.load_state_dict(loaded_dict["critic_state_dict"])
        if "depth_encoder_state_dict" in loaded_dict and loaded_dict["depth_encoder_state_dict"] is not None:
            self.alg.actor_critic.depth_encoder.load_state_dict(loaded_dict["depth_encoder_state_dict"])
        if "depth_actor_state_dict" in loaded_dict and loaded_dict["depth_actor_state_dict"] is not None:
            self.alg.actor_critic.depth_actor.load_state_dict(loaded_dict["depth_actor_state_dict"])
        # -- Load observation normalizer if used
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
        # -- Load optimizer if used
        if load_optimizer:
            # -- PPO
            self.alg.ac_optimizer.load_state_dict(loaded_dict["ac_optimizer_state_dict"])
            if "depth_encoder_optimizer_state_dict" in loaded_dict and loaded_dict["depth_encoder_optimizer_state_dict"] is not None:
                self.alg.depth_encoder_optimizer.load_state_dict(loaded_dict["depth_encoder_optimizer_state_dict"])
            if "depth_actor_optimizer_state_dict" in loaded_dict and loaded_dict["depth_actor_optimizer_state_dict"] is not None:
                self.alg.depth_actor_optimizer.load_state_dict(loaded_dict["depth_actor_optimizer_state_dict"])

        # -- Load current learning iteration
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]


    def get_inference_policy_scan(self, device=None):
        self.eval_mode()
        if device is not None:
            self.alg.actor_critic.to(device)

        def policy(observations, scan_observations, history_observations=None, hist_encoding=None):
            if self.cfg["empirical_normalization"]:
                if device is not None:
                    self.obs_normalizer.to(device)
                observations = self.obs_normalizer(observations)
                scan_observations = self.scan_obs_normalizer(scan_observations)

            return self.alg.actor_critic.act_inference(observations, scan_observations,)

        return policy


    def get_inference_policy_vision(self, device=None):
        self.eval_mode()
        if device is not None:
            self.alg.actor_critic.to(device)

        def policy(observations, vision_obs_batch, history_observations=None, hist_encoding=None):
            if self.cfg["empirical_normalization"]:
                if device is not None:
                    self.obs_normalizer.to(device)
                observations = self.obs_normalizer(observations)
                vision_obs_batch = self.vision_obs_normalizer(vision_obs_batch)

            return self.alg.actor_critic.depth_act(observations, vision_obs_batch,)

        return policy


    def scan_train_mode(self):
        # -- PPO
        self.alg.actor_critic.train()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def vision_train_mode(self):
        # -- PPO
        self.alg.actor_critic.depth_actor.train()
        self.alg.actor_critic.depth_encoder.train()
        self.alg.actor_critic.actor.eval()
        self.alg.actor_critic.critic.eval()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        # -- PPO
        self.alg.actor_critic.eval()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)
