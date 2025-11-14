# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import torch

import carb

import isaaclab.utils.math as math_utils


def sub_keyboard_event(event, cmd_vel, lin_vel=1.0, ang_vel=1.0) -> bool:
    """
    This function is subscribed to keyboard events and updates the velocity commands.
    """
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        # Update the velocity commands for the first environment (0th index)
        if event.input.name == "UP":
            cmd_vel[0] += torch.tensor([lin_vel, 0, 0], dtype=torch.float32)
        elif event.input.name == "DOWN":
            cmd_vel[0] += torch.tensor([-lin_vel, 0, 0], dtype=torch.float32)
        elif event.input.name == "LEFT":
            cmd_vel[0] += torch.tensor([0, lin_vel, 0], dtype=torch.float32)
        elif event.input.name == "RIGHT":
            cmd_vel[0] += torch.tensor([0, -lin_vel, 0], dtype=torch.float32)
        elif event.input.name == "J":
            cmd_vel[0] += torch.tensor([0, 0, ang_vel], dtype=torch.float32)
        elif event.input.name == "L":
            cmd_vel[0] += torch.tensor([0, 0, -ang_vel], dtype=torch.float32)
        elif event.input.name == "R":
            cmd_vel.zero_()
        print("Cmd Now( Vx: ",format(cmd_vel[0,0].item(), '.1f'),",  Vy: ",format(cmd_vel[0,1].item(), '.1f'),",  Wz: ",format(cmd_vel[0,2].item(), '.1f'),") ")

    # Reset velocity commands on key release
    # elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
    #     cmd_vel.zero_()

    return True

def sub_keyboard_event_6d(event, cmd_vel, lin_vel=1.0, ang_vel=1.0) -> bool:
    """
    This function is subscribed to keyboard events and updates the velocity commands.
    """
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        # Update the velocity commands for the first environment (0th index)
        if event.input.name == "UP":
            cmd_vel[0] += torch.tensor([lin_vel, 0, 0, 0, 0, 0], dtype=torch.float32)
        elif event.input.name == "DOWN":
            cmd_vel[0] += torch.tensor([-lin_vel, 0, 0, 0, 0, 0], dtype=torch.float32)
        elif event.input.name == "LEFT":
            cmd_vel[0] += torch.tensor([0, lin_vel, 0, 0, 0, 0], dtype=torch.float32)
        elif event.input.name == "RIGHT":
            cmd_vel[0] += torch.tensor([0, -lin_vel, 0, 0, 0, 0], dtype=torch.float32)
        elif event.input.name == "J":
            cmd_vel[0] += torch.tensor([0, 0, ang_vel, 0, 0, 0], dtype=torch.float32)
        elif event.input.name == "L":
            cmd_vel[0] += torch.tensor([0, 0, -ang_vel, 0, 0, 0], dtype=torch.float32)
        elif event.input.name == "U":
            cmd_vel[0] += torch.tensor([0, 0, 0, 0.05, 0, 0], dtype=torch.float32)
        elif event.input.name == "I":
            cmd_vel[0] += torch.tensor([0, 0, 0, -0.05, 0, 0], dtype=torch.float32)
        elif event.input.name == "O":
            cmd_vel[0] += torch.tensor([0, 0, 0, 0, 0.05, 0], dtype=torch.float32)
        elif event.input.name == "P":
            cmd_vel[0] += torch.tensor([0, 0, 0, 0, -0.05, 0], dtype=torch.float32)
        elif event.input.name == "N":
            cmd_vel[0] += torch.tensor([0, 0, 0, 0, 0, 0.05], dtype=torch.float32)
        elif event.input.name == "M":
            cmd_vel[0] += torch.tensor([0, 0, 0, 0, 0, -0.05], dtype=torch.float32)
        elif event.input.name == "R":
            cmd_vel.zero_()
        print("Cmd Now( Vx: ",format(cmd_vel[0,0].item(), '.1f'),",  Vy: ",format(cmd_vel[0,1].item(), '.1f'),",  Wz: ",
              format(cmd_vel[0,2].item(), '.1f'),",  Roll: ",format(cmd_vel[0,3].item(),'.1f'),",  Pitch: ",format(cmd_vel[0,4].item(),'.1f'),",  Height: ",format(cmd_vel[0,5].item(),'.1f'),") ")

    # Reset velocity commands on key release
    # elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
    #     cmd_vel.zero_()

    return True

def sub_keyboard_event_stand(event, cmd_vel, lin_vel=1.0, ang_vel=1.0) -> bool:
    """
    This function is subscribed to keyboard events and updates the velocity commands.
    """
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        # Update the velocity commands for the first environment (0th index)
        if event.input.name == "UP":
            cmd_vel[0] += torch.tensor([lin_vel, 0, 0, 0], dtype=torch.float32)
        elif event.input.name == "DOWN":
            cmd_vel[0] += torch.tensor([-lin_vel, 0, 0, 0], dtype=torch.float32)
        elif event.input.name == "LEFT":
            cmd_vel[0] += torch.tensor([0, lin_vel, 0, 0], dtype=torch.float32)
        elif event.input.name == "RIGHT":
            cmd_vel[0] += torch.tensor([0, -lin_vel, 0, 0], dtype=torch.float32)
        elif event.input.name == "J":
            cmd_vel[0] += torch.tensor([0, 0, ang_vel, 0], dtype=torch.float32)
        elif event.input.name == "L":
            cmd_vel[0] += torch.tensor([0, 0, -ang_vel, 0], dtype=torch.float32)
        elif event.input.name == "KEY_1":
            cmd_vel[0, 3] = 1
        elif event.input.name == "KEY_0":
            cmd_vel[0, 3] = 0
        elif event.input.name == "R":
            cmd_vel.zero_()
        print("Cmd Now( Vx: ",format(cmd_vel[0,0].item(), '.1f'),",  Vy: ",format(cmd_vel[0,1].item(), '.1f'),",  Wz: ",
              format(cmd_vel[0,2].item(), '.1f'),",  key: ",format(cmd_vel[0,3].item(),'.1f'),") ")

    # Reset velocity commands on key release
    # elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
    #     cmd_vel.zero_()

    return True


def sub_keyboard_event_position(event, cmd_pos, mode_boolean, pos_x=1.0, pos_y=1.0) -> bool:
    """
    This function is subscribed to keyboard events and updates the velocity commands.
    """
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        # Update the velocity commands for the first environment (0th index)
        if event.input.name == "UP":
            cmd_pos[0] += torch.tensor([pos_x, 0, 0], dtype=torch.float32)
        elif event.input.name == "DOWN":
            cmd_pos[0] += torch.tensor([-pos_x, 0, 0], dtype=torch.float32)
        elif event.input.name == "LEFT":
            cmd_pos[0] += torch.tensor([0, pos_y, 0], dtype=torch.float32)
        elif event.input.name == "RIGHT":
            cmd_pos[0] += torch.tensor([0, -pos_y, 0], dtype=torch.float32)
        elif event.input.name == "KEY_1":
            mode_boolean[0, 0] = 1
        elif event.input.name == "KEY_0":
            mode_boolean[0, 0] = 0
        elif event.input.name == "R":
            cmd_pos.zero_()
        print("Cmd Now( x: ",format(cmd_pos[0,0].item(), '.1f'),",  y: ",format(cmd_pos[0,1].item(), '.1f'),",  mode: ",format(mode_boolean[0,0].item(), '.1f'),") ")

    # Reset velocity commands on key release
    # elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
    #     cmd_vel.zero_()

    return True


def camera_follow(env):
    if not hasattr(camera_follow, "smooth_camera_positions"):
        camera_follow.smooth_camera_positions = []
    robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
    robot_quat = env.unwrapped.scene["robot"].data.root_quat_w[0]
    camera_offset = torch.tensor([-5.0, 0.0, 1.0], dtype=torch.float32, device=env.device)
    camera_pos = math_utils.transform_points(
        camera_offset.unsqueeze(0), pos=robot_pos.unsqueeze(0), quat=robot_quat.unsqueeze(0)
    ).squeeze(0)
    camera_pos[2] = torch.clamp(camera_pos[2], min=0.0)
    window_size = 50
    camera_follow.smooth_camera_positions.append(camera_pos)
    if len(camera_follow.smooth_camera_positions) > window_size:
        camera_follow.smooth_camera_positions.pop(0)
    smooth_camera_pos = torch.mean(torch.stack(camera_follow.smooth_camera_positions), dim=0)
    env.unwrapped.viewport_camera_controller.set_view_env_index(env_index=0)
    env.unwrapped.viewport_camera_controller.update_view_location(
        eye=smooth_camera_pos.cpu().numpy(), lookat=robot_pos.cpu().numpy()
    )
