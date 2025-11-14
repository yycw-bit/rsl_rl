# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""
from sympy import false

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument("--MLP", action="store_false", help="If use MLP net.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import torch

import carb
import omni
import rsl_rl_utils
from rsl_rl.runners import OnPolicyRunner, OnPolicyRunnerWithEstimator

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from rsl_rl.env import RslRlVecEnvWrapperBIT
import robot_lab.tasks  # noqa: F401
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from plotter import Logger


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # make a smaller scene for play
    env_cfg.scene.num_envs = 64
    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 4
        env_cfg.scene.terrain.terrain_generator.num_cols = 8
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None

    env_cfg.commands.base_velocity.heading_command = false
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (-0.8, 0.8)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (-0.8, 0.8)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
    # env_cfg.commands.base_velocity.ranges.heading = (-0.1, 0.1)
    # env_cfg.commands.base_velocity.ranges.lin_vel_x = (0., 0.)
    # env_cfg.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
    # env_cfg.commands.base_velocity.ranges.ang_vel_z = (-.0, .0)
    # env_cfg.commands.base_velocity.ranges.heading = (-0., 0.)

    env_cfg.terminations.illegal_contact = None

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        cmd_vel = torch.zeros((env_cfg.scene.num_envs, 4), dtype=torch.float32)
        system_input = carb.input.acquire_input_interface()
        system_input.subscribe_to_keyboard_events(
            omni.appwindow.get_default_app_window().get_keyboard(),
            lambda event: rsl_rl_utils.sub_keyboard_event_stand(event, cmd_vel, lin_vel=0.1, ang_vel=0.1),
        )
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: cmd_vel.clone().to(env.device),
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapperBIT(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if  args_cli.MLP:
        ppo_runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
        )
        print("MLP")
    else:
        ppo_runner = OnPolicyRunnerWithEstimator(
            env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
        )
        print("Estimator")
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_onnx(
    #     actor_critic=ppo_runner.alg.actor_critic,
    #     normalizer=ppo_runner.obs_normalizer,
    #     path=export_model_dir,
    #     filename="policy.onnx",
    # )
    export_policy_as_jit(
        policy=ppo_runner.alg.actor_critic,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.pt",
    )

    dt = env.unwrapped.physics_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0

    logger = Logger(dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 12  # which joint is used for logging
    stop_state_log =  2*int(env.max_episode_length) - 1  # 100 # number of steps before plotting states
    # stop_rew_log = 10 * env.max_episode_length + 1  # number of steps before print average episode rewards
    i =0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # actions = torch.zeros_like(actions)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # if args_cli.keyboard:
        #     rsl_rl_utils.camera_follow(env)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

        # if i < stop_state_log:
        #     logger.log_states(
        #         {
        #             'dof_pos_target': env_cfg.actions.joint_vel.scale * actions[robot_index, joint_index].item() ,
        #             'dof_pos': env.unwrapped.scene["robot"].data.joint_pos[robot_index, joint_index].item(),
        #             'dof_vel': env.unwrapped.scene["robot"].data.joint_vel[robot_index, joint_index].item(),
        #             # 'dof_torque_0': env.unwrapped.scene["robot"].data.applied_torque[robot_index, joint_index].item(),
        #             # 'dof_torque_1': env.unwrapped.scene["robot"].data.applied_torque[robot_index, joint_index + 1].item(),
        #             # 'dof_torque_2': env.unwrapped.scene["robot"].data.applied_torque[robot_index, joint_index + 2].item(),
        #             # 'dof_torque_3': env.unwrapped.scene["robot"].data.applied_torque[robot_index, joint_index + 3].item(),
        #             # 'dof_torque_4': env.unwrapped.scene["robot"].data.applied_torque[robot_index, joint_index + 4].item(),
        #             # 'dof_torque_5': env.unwrapped.scene["robot"].data.applied_torque[robot_index, joint_index + 5].item(),
        #             # 'dof_torque_6': env.unwrapped.scene["robot"].data.applied_torque[robot_index, joint_index + 6].item(),
        #             # 'dof_torque_7': env.unwrapped.scene["robot"].data.applied_torque[robot_index, joint_index + 7].item(),
        #             # 'dof_torque_8': env.unwrapped.scene["robot"].data.applied_torque[robot_index, joint_index + 8].item(),
        #             # 'dof_torque_9': env.unwrapped.scene["robot"].data.applied_torque[robot_index, joint_index + 9].item(),
        #             # 'dof_torque_10': env.unwrapped.scene["robot"].data.applied_torque[robot_index, joint_index + 10].item(),
        #             # 'dof_torque_11': env.unwrapped.scene["robot"].data.applied_torque[robot_index, joint_index + 11].item(),
        #             'command_x': env.unwrapped.command_manager.get_command("base_velocity")[robot_index, 0].item(),
        #             'command_y': env.unwrapped.command_manager.get_command("base_velocity")[robot_index, 1].item(),
        #             'command_yaw': env.unwrapped.command_manager.get_command("base_velocity")[robot_index, 2].item(),
        #             'base_vel_x': env.unwrapped.scene["robot"].data.root_com_lin_vel_b[robot_index, 0].item(),
        #             'base_vel_y': env.unwrapped.scene["robot"].data.root_com_lin_vel_b[robot_index, 1].item(),
        #             'base_vel_z': env.unwrapped.scene["robot"].data.root_com_lin_vel_b[robot_index, 2].item(),
        #             'base_vel_yaw': env.unwrapped.scene["robot"].data.root_com_ang_vel_b[robot_index, 2].item(),
        #             'contact_forces_z':env.unwrapped.scene["contact_forces"].data.net_forces_w[robot_index, 2].cpu().numpy(),
        #         }
        #     )
        # elif i == stop_state_log:
        #     logger.plot_states()
        #     # if  0 < i < stop_rew_log:
        #     #     if infos["episode"]:
        #     #         num_episodes = torch.sum(env.reset_buf).item()
        #     #         if num_episodes>0:
        #     #             logger.log_rewards(infos["episode"], num_episodes)
        # i = i+1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
