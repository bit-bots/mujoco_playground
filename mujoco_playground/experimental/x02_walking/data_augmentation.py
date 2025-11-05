import torch
from typing import Optional, Tuple, Any

def get_symmetric_states_x02(
    obs: Optional[torch.Tensor] = None,
    actions: Optional[torch.Tensor] = None,
    cfg: "BaseEnvCfg" = None,
    obs_type: str = "policy",
    env: Any = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mirror observations and actions for X02 humanoid (left/right symmetry).
    
    Observation structure (from joystick.py):
    - qvel_history: 10*history_len (joint velocities)
    - qpos_error_history: 10*history_len (joint position errors)
    - gyro: 3
    - gravity: 3
    - command: 3 [lin_vel_x, lin_vel_y, ang_vel_yaw]
    - joint_angles: 10 (L_hip_yaw, L_hip_roll, L_hip_pitch, L_knee_pitch, L_ankle, 
                        R_hip_yaw, R_hip_roll, R_hip_pitch, R_knee_pitch, R_ankle)
    - joint_vel: 10 (same order as joint_angles)
    - last_action: 10 (actuators, same as joint_angles)
    - phase: 2 [cos, sin]
    
    Joint order (indices 0-9):
    Left: 0=L_hip_yaw, 1=L_hip_roll, 2=L_hip_pitch, 3=L_knee_pitch, 4=L_ankle
    Right: 5=R_hip_yaw, 6=R_hip_roll, 7=R_hip_pitch, 8=R_knee_pitch, 9=R_ankle
    """
    if obs is None and actions is None:
        return obs, actions
    
    # Get history length from config or default
    history_len = cfg.history_len if cfg is not None and hasattr(cfg, 'history_len') else 1
    
    if obs is not None:
        obs = obs.clone()
        obs_dim = obs.shape[-1]
        
        # Calculate dimension offsets
        qvel_history_dim = 10 * history_len
        qpos_error_history_dim = 10 * history_len
        gyro_dim = 3
        gravity_dim = 3
        command_dim = 3
        joint_angles_dim = 10
        joint_vel_dim = 10
        last_action_dim = 10
        phase_dim = 2
        
        # Expected total dimension
        expected_dim = (qvel_history_dim + qpos_error_history_dim + gyro_dim + 
                       gravity_dim + command_dim + joint_angles_dim + 
                       joint_vel_dim + last_action_dim + phase_dim)
        
        if obs_dim != expected_dim:
            raise ValueError(f"Observation dimension {obs_dim} doesn't match expected {expected_dim}. "
                           f"History len: {history_len}")
        
        # Create indices for slicing
        idx = 0
        qvel_history_start = idx
        qvel_history_end = idx + qvel_history_dim
        idx = qvel_history_end
        
        qpos_error_history_start = idx
        qpos_error_history_end = idx + qpos_error_history_dim
        idx = qpos_error_history_end
        
        gyro_start = idx
        gyro_end = idx + gyro_dim
        idx = gyro_end
        
        gravity_start = idx
        gravity_end = idx + gravity_dim
        idx = gravity_end
        
        command_start = idx
        command_end = idx + command_dim
        idx = command_end
        
        joint_angles_start = idx
        joint_angles_end = idx + joint_angles_dim
        idx = joint_angles_end
        
        joint_vel_start = idx
        joint_vel_end = idx + joint_vel_dim
        idx = joint_vel_end
        
        last_action_start = idx
        last_action_end = idx + last_action_dim
        idx = last_action_end
        
        phase_start = idx
        phase_end = idx + phase_dim
        
        # Mirror qvel_history (swap left/right legs)
        qvel_history = obs[..., qvel_history_start:qvel_history_end]
        qvel_history = qvel_history.reshape(-1, history_len, 10)  # Reshape to (..., history_len, 10)
        # Swap left (0-4) and right (5-9) legs
        qvel_history_mirrored = qvel_history.clone()
        qvel_history_mirrored[..., :, [0, 1, 2, 3, 4]] = qvel_history[..., :, [5, 6, 7, 8, 9]]  # Right -> Left
        qvel_history_mirrored[..., :, [5, 6, 7, 8, 9]] = qvel_history[..., :, [0, 1, 2, 3, 4]]  # Left -> Right
        qvel_history_mirrored = qvel_history_mirrored.reshape(-1, qvel_history_dim)
        obs[..., qvel_history_start:qvel_history_end] = qvel_history_mirrored
        
        # Mirror qpos_error_history (same as qvel_history)
        qpos_error_history = obs[..., qpos_error_history_start:qpos_error_history_end]
        qpos_error_history = qpos_error_history.reshape(-1, history_len, 10)
        qpos_error_history_mirrored = qpos_error_history.clone()
        qpos_error_history_mirrored[..., :, [0, 1, 2, 3, 4]] = qpos_error_history[..., :, [5, 6, 7, 8, 9]]
        qpos_error_history_mirrored[..., :, [5, 6, 7, 8, 9]] = qpos_error_history[..., :, [0, 1, 2, 3, 4]]
        qpos_error_history_mirrored = qpos_error_history_mirrored.reshape(-1, qpos_error_history_dim)
        obs[..., qpos_error_history_start:qpos_error_history_end] = qpos_error_history_mirrored
        
        # Mirror gyro: flip x and z, keep y
        obs[..., gyro_start:gyro_end] = torch.stack([
            -obs[..., gyro_start + 0],
            obs[..., gyro_start + 1],
            -obs[..., gyro_start + 2],
        ], dim=-1)
        
        # Mirror gravity: flip y, keep x and z
        obs[..., gravity_start:gravity_end] = torch.stack([
            obs[..., gravity_start + 0],
            -obs[..., gravity_start + 1],
            obs[..., gravity_start + 2],
        ], dim=-1)
        
        # Mirror command: flip lin_vel_y, negate ang_vel_yaw
        obs[..., command_start:command_end] = torch.stack([
            obs[..., command_start + 0],
            -obs[..., command_start + 1],
            -obs[..., command_start + 2],
        ], dim=-1)
        
        # Mirror joint angles (swap left/right)
        joint_angles = obs[..., joint_angles_start:joint_angles_end]
        joint_angles_mirrored = joint_angles.clone()
        joint_angles_mirrored[..., [0, 1, 2, 3, 4]] = joint_angles[..., [5, 6, 7, 8, 9]]  # Right -> Left
        joint_angles_mirrored[..., [5, 6, 7, 8, 9]] = joint_angles[..., [0, 1, 2, 3, 4]]  # Left -> Right
        obs[..., joint_angles_start:joint_angles_end] = joint_angles_mirrored
        
        # Mirror joint velocities (same as joint angles)
        joint_vel = obs[..., joint_vel_start:joint_vel_end]
        joint_vel_mirrored = joint_vel.clone()
        joint_vel_mirrored[..., [0, 1, 2, 3, 4]] = joint_vel[..., [5, 6, 7, 8, 9]]
        joint_vel_mirrored[..., [5, 6, 7, 8, 9]] = joint_vel[..., [0, 1, 2, 3, 4]]
        obs[..., joint_vel_start:joint_vel_end] = joint_vel_mirrored
        
        # Mirror last action (swap left/right leg actuators)
        last_action = obs[..., last_action_start:last_action_end]
        last_action_mirrored = last_action.clone()
        last_action_mirrored[..., [0, 1, 2, 3, 4]] = last_action[..., [5, 6, 7, 8, 9]]  # Right -> Left
        last_action_mirrored[..., [5, 6, 7, 8, 9]] = last_action[..., [0, 1, 2, 3, 4]]  # Left -> Right
        obs[..., last_action_start:last_action_end] = last_action_mirrored
        
        # Phase doesn't need mirroring (it's symmetric)
        # obs[..., phase_start:phase_end] remains unchanged
        
    if actions is not None:
        actions = actions.clone()
        action_dim = actions.shape[-1]
        
        # Mirror actions (swap left/right leg actuators)
        if action_dim == 10:
            actions_mirrored = actions.clone()
            actions_mirrored[..., [0, 1, 2, 3, 4]] = actions[..., [5, 6, 7, 8, 9]]  # Right -> Left
            actions_mirrored[..., [5, 6, 7, 8, 9]] = actions[..., [0, 1, 2, 3, 4]]  # Left -> Right
            actions = actions_mirrored
        else:
            raise ValueError(f"Action dimension {action_dim} not supported. Expected 10.")
    
    return obs, actions
