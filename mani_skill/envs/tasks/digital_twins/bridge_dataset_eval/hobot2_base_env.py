from typing import Literal

import numpy as np
import sapien
from hobot.robot_common.widowx250s_constants import WX_TELEOP_RANGE

from mani_skill.agents.controllers.pd_joint_pos import (
    PDJointPosControllerConfig,
    PDJointPosMimicControllerConfig,
)
from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval.base_env import (
    BaseBridgeEnv,
    WidowX250SBridgeDatasetFlatTable,
)
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs.types import SimConfig


class Hobot2BridgeFlatTable(WidowX250SBridgeDatasetFlatTable):
    JOINT_LOWER = WX_TELEOP_RANGE[:6, 0]
    JOINT_UPPER = WX_TELEOP_RANGE[:6, 1]

    @property
    def _sensor_configs(self):

        # TODO(andrew): make this alf.configurable
        scale = 3
        width = 228 * scale
        height = 128 * scale
        intrinsic = np.array([[124.409, 0, 114.0], [0, 124.409, 64.0], [0, 0, 1]])
        intrinsic[:2] *= scale

        return super()._sensor_configs + [
            CameraConfig(
                uid="external_camera",
                # the camera used for real evaluation for the sink setup
                pose=sapien.Pose(
                    [0.80, 0.0, 0.1],
                    [0.0, 0.0, 0.0, 1.0],
                ),
                entity_uid="base_link",
                width=width,
                height=height,
                near=0.01,
                far=2.0,
                intrinsic=intrinsic,
            ),
        ]

    @property
    def _controller_configs(self):
        arm_common_kwargs = dict(
            joint_names=self.arm_joint_names,
            lower=self.JOINT_LOWER,
            upper=self.JOINT_UPPER,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            use_delta=False,
            interpolate=True,
            normalize_action=False,
        )
        arm_pd_absolute_joint_target = PDJointPosControllerConfig(**arm_common_kwargs)

        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            joint_names=self.gripper_joint_names,
            lower=-0.04,
            upper=0.04,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            normalize_action=True,
            drive_mode="force",
        )
        controller = dict(
            arm=arm_pd_absolute_joint_target, gripper=gripper_pd_joint_pos
        )
        return dict(arm_pd_absolute_joint_target_gripper_pd_joint_pos=controller)


class Hobot2BridgeEnv(BaseBridgeEnv):
    SUPPORTED_OBS_MODES = ["rgb+depth+segmentation"]
    scene_setting: Literal["hobot2_flat_table"] = "hobot2_flat_table"

    def __init__(self, **kwargs):
        kwargs["robot_cls"] = Hobot2BridgeFlatTable
        super().__init__(**kwargs)

    # In hobot2, we use 10Hz control frequency
    # We'll also use a sim timestep of 5ms
    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=200, control_freq=10, spacing=20)
