import torch
import tensordict
from typing import Optional, Tuple, Any

def get_symmetric_states_x02(
    obs: tensordict.TensorDict = None,
    actions: Optional[torch.Tensor] = None,
    cfg: "BaseEnvCfg" = None,
    obs_type: str = "policy",
    env: Any = None,
) -> Tuple[tensordict.TensorDict, torch.Tensor]:
    return obs, actions


def get_symmetric_states_berkeley(
    obs: tensordict.TensorDict = None,
    actions: Optional[torch.Tensor] = None,
    cfg: "BaseEnvCfg" = None,
    obs_type: str = "policy",
    env: Any = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if obs is None and actions is None:
        return obs, actions

    def leg_mirror(mir_obs, obs, start_idx, mir_type="state"):
        if mir_type == "state" or mir_type == "privileged_state":
            m = mir_obs[mir_type]
            o = obs[mir_type]
        elif mir_type == "action":
            m = mir_obs
            o = obs
        else:
            raise ValueError(f"Invalid mirror type: {mir_type}")
        
        m[:,start_idx] = -1 * o[:,start_idx + 6] # l_hip_roll = -1 * r_hip_roll
        m[:,start_idx + 1] = -1 * o[:,start_idx + 7] # l_hip_yaw = -1 * r_hip_yaw
        m[:,start_idx + 2] = +1 * o[:,start_idx + 8] # l_hip_pitch = +1 * r_hip_pitch
        m[:,start_idx + 3] = +1 * o[:,start_idx + 9] # l_knee_pitch = +1 * r_knee_pitch
        m[:,start_idx + 4] = +1 * o[:,start_idx + 10] # l_ankle_pitch = +1 * r_ankle_pitch
        m[:,start_idx + 5] = -1 * o[:,start_idx + 11] # l_ankle_roll = -1 * r_ankle_roll
        m[:,start_idx + 6] = -1 * o[:,start_idx] # r_hip_roll = -1 * l_hip_roll
        m[:,start_idx + 7] = -1 * o[:,start_idx + 1] # r_hip_yaw = -1 * l_hip_yaw
        m[:,start_idx + 8] = +1 * o[:,start_idx + 2] # r_hip_pitch = +1 * l_hip_pitch
        m[:,start_idx + 9] = +1 * o[:,start_idx + 3] # r_knee_pitch = +1 * l_knee_pitch
        m[:,start_idx + 10] = +1 * o[:,start_idx + 4] # r_ankle_pitch = +1 * l_ankle_pitch
        m[:,start_idx + 11] = -1 * o[:,start_idx + 5] # r_ankle_roll = -1 * l_ankle_roll

    if obs is not None:
        mir_obs = obs.clone()
        mir_obs["state"][:,0] *= +1 # x component of linvel
        mir_obs["state"][:,1] *= -1 # y component of linvel
        mir_obs["state"][:,2] *= +1 # z component of linvel

        mir_obs["state"][:,3] *= -1 # x component of gyro
        mir_obs["state"][:,4] *= +1 # y component of gyro
        mir_obs["state"][:,5] *= -1 # z component of gyro

        mir_obs["state"][:,6] *= +1 # x component of gravity
        mir_obs["state"][:,7] *= -1 # y component of gravity
        mir_obs["state"][:,8] *= +1 # z component of gravity

        mir_obs["state"][:,9] *= +1 # x component of command
        mir_obs["state"][:,10] *= -1 # y component of command
        mir_obs["state"][:,11] *= +1 # z component of command
        # mirror joint angles
        leg_mirror(mir_obs, obs, 12, mir_type="state")
        # mirror joint velocities (same as joint angles)
        leg_mirror(mir_obs, obs, 24, mir_type="state")
        # mirror last action (same as joint angles)
        leg_mirror(mir_obs, obs, 36, mir_type="state")
        # swap left and right leg phases
        mir_obs["state"][:,48:49] = obs["state"][:,50:51] # phase = phase
        mir_obs["state"][:,50:51] = obs["state"][:,48:49] # phase = phase
         # first part of privileged state is the state
        mir_obs["privileged_state"][:, :52] = mir_obs["state"].clone()
        # true gyro
        mir_obs["privileged_state"][:,52] *= -1 # x component of gyro
        mir_obs["privileged_state"][:,53] *= +1 # y component of gyro
        mir_obs["privileged_state"][:,54] *= -1 # z component of gyro
        # accelerometer
        mir_obs["privileged_state"][:,55] *= +1 # x component of accelerometer
        mir_obs["privileged_state"][:,56] *= -1 # y component of accelerometer
        mir_obs["privileged_state"][:,57] *= +1 # z component of accelerometer
        # true gravity
        mir_obs["privileged_state"][:,58] *= +1 # x component of gravity
        mir_obs["privileged_state"][:,59] *= -1 # y component of gravity
        mir_obs["privileged_state"][:,60] *= +1 # z component of gravity
        # true linvel
        mir_obs["privileged_state"][:,61] *= +1 # x component of linvel
        mir_obs["privileged_state"][:,62] *= -1 # y component of linvel
        mir_obs["privileged_state"][:,63] *= +1 # z component of linvel
        # true global_angvel
        mir_obs["privileged_state"][:,64] *= +1 # x component of global_angvel honestly not sure what to do with this
        mir_obs["privileged_state"][:,65] *= +1 # y component of global_angvel honestly not sure what to do with this
        mir_obs["privileged_state"][:,66] *= +1 # z component of global_angvel honestly not sure what to do with this
        # true joint angles
        leg_mirror(mir_obs, obs, 67, mir_type="privileged_state")
        # true joint velocities (same as joint angles)
        leg_mirror(mir_obs, obs, 79, mir_type="privileged_state")
        # true root height
        mir_obs["privileged_state"][:,91] *= +1 # root height (no change)
        # true actuator forces
        leg_mirror(mir_obs, obs, 92, mir_type="privileged_state")
        # swap foot contacts 
        mir_obs["privileged_state"][:,104] = obs["privileged_state"][:,105]
        mir_obs["privileged_state"][:,105] = obs["privileged_state"][:,104]
        # feet velocities
        mir_obs["privileged_state"][:,106:109] = obs["privileged_state"][:,109:112]
        mir_obs["privileged_state"][:,109:112] = obs["privileged_state"][:,106:109]
        # feet air time
        mir_obs["privileged_state"][:,112] = obs["privileged_state"][:,113]
        mir_obs["privileged_state"][:,113] = obs["privileged_state"][:,112]
        # concat obs and mir obs
        obs_aug = torch.cat([obs, mir_obs], dim=0)
    else:
        obs_aug = None
    
    if actions is not None:
        mir_actions = actions.clone()
        leg_mirror(mir_actions, actions, 0, mir_type="action")
        actions_aug = torch.cat([actions, mir_actions], dim=0)
    else:
        actions_aug = None
    #if obs is not None and mir_obs is not None:
    #    print(f"obs0: {obs[0]}")
    #    print(f"mir_obs0: {mir_obs[0]}")
    #else:
    #    print("No obs or mir_obs")
    #if actions is not None and mir_actions is not None:
    #    print(f"actions0: {actions[0]}")
    #     print(f"mir_actions0: {mir_actions[0]}")
    #else:
    #    print("No actions or mir_actions")
    return obs_aug, actions_aug