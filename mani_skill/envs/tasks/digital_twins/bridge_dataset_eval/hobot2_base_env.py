from typing import Literal

import numpy as np
import sapien
from hobot.robot_common.widowx250s_constants import WX_TELEOP_RANGE

from mani_skill.agents.controllers.pd_joint_pos import (
    PDJointPosControllerConfig,
    PDJointPosMimicControllerConfig,
)
from mani_skill.envs.scene import ManiSkillScene
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
                    [0.80, 0.0, 0.23],
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
            stiffness=np.array(self.arm_stiffness)*2.0,
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


def create_hobot2_boundary(scene: ManiSkillScene,
                           pos: np.ndarray = np.array([0, 0, 0]),
                           quat: np.ndarray = np.array([1, 0, 0, 0]),
                           outer_size: np.ndarray = np.array([0.2, 0.2, 0.2]),
                           wall_thickness: float = 0.005,
                           color: np.ndarray = np.array([0.8, 0.8, 0.8, 1]),
                           name: str = "hobot2_boundary",
                           visual: bool = False,
                           collision: bool = True,
                           slope: float = 0.0):
    """
    Create a hollow boundary with four sides (no top/bottom) in SAPIEN.
    Can provide a non-zero slope to create a tray.

    Args:
        scene: SAPIEN scene object
        pos: Position [x, y, z] of the center of the box
        quat: Quaternion [w, x, y, z] for orientation
        outer_size: Outer dimensions [length, width, height] of the box
        wall_thickness: Thickness of the walls
        color: RGBA color of the box
        name: Name of the actor
        visual: Whether to create visual geometries
        collision: Whether to create collision geometries
        slope: Wall outward slope in radians (0 = vertical walls)

    Returns:
        The created actor
    """
    builder = scene.create_actor_builder()

    length, width, height = outer_size
    half_length, half_width, half_height = length / 2, width / 2, height / 2
    half_thick = wall_thickness / 2

    length_scale = 1

    # If there is a slope, we want to extend the lengths to cover the corners.
    if slope != 0:
        length_scale = 2

    visual_material = sapien.render.RenderMaterial()
    visual_material.set_base_color(color)

    # Calculate actual wall length accounting for the slope (hypotenuse)
    wall_height = height / np.cos(abs(slope))
    half_wall_height = wall_height / 2

    # Helper function to create quaternion from axis-angle
    def axis_angle_to_quat(axis, angle):
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)  # Normalize axis

        sin_half = np.sin(angle / 2)
        cos_half = np.cos(angle / 2)

        x = axis[0] * sin_half
        y = axis[1] * sin_half
        z = axis[2] * sin_half
        w = cos_half

        return [w, x, y, z]  # SAPIEN quaternion format is [w, x, y, z]

    # For rotation around bottom edge, we need to:
    # 1. Calculate the rotation quaternion
    # 2. Move the pivot point to the bottom edge
    # 3. Apply the rotation
    # 4. Move the pivot point back

    # Wall 1 (along x-axis, positive y) - rotation around x-axis at bottom edge
    x_axis = [1, 0, 0]
    wall1_rot = axis_angle_to_quat(x_axis, slope)

    # Calculate the position after rotation around bottom edge
    # The wall pivots around its bottom edge, which is at z = -half_height
    # After rotation, the center moves up and slightly inward
    wall1_pos_z = -half_height + half_wall_height * np.cos(slope)
    wall1_pos_y = half_width - half_thick - half_wall_height * np.sin(slope)
    wall1_pos = [0, wall1_pos_y, wall1_pos_z]
    wall1_size = [half_length*length_scale, half_thick, half_wall_height]
    wall1_pose = sapien.Pose(wall1_pos, wall1_rot)

    # Wall 2 (along x-axis, negative y) - rotation around x-axis at bottom edge
    wall2_rot = axis_angle_to_quat(x_axis, -slope)
    wall2_pos_z = -half_height + half_wall_height * np.cos(slope)
    wall2_pos_y = -half_width + half_thick + half_wall_height * np.sin(slope)
    wall2_pos = [0, wall2_pos_y, wall2_pos_z]
    wall2_size = [half_length*length_scale, half_thick, half_wall_height]
    wall2_pose = sapien.Pose(wall2_pos, wall2_rot)

    # Wall 3 (along y-axis, positive x) - rotation around y-axis at bottom edge
    y_axis = [0, 1, 0]
    wall3_rot = axis_angle_to_quat(y_axis, -slope)
    wall3_pos_z = -half_height + half_wall_height * np.cos(slope)
    wall3_pos_x = half_length - half_thick - half_wall_height * np.sin(slope)
    wall3_pos = [wall3_pos_x, 0, wall3_pos_z]
    wall3_size = [half_thick, (half_width - wall_thickness)*length_scale,
                  half_wall_height]
    wall3_pose = sapien.Pose(wall3_pos, wall3_rot)

    # Wall 4 (along y-axis, negative x) - rotation around y-axis at bottom edge
    wall4_rot = axis_angle_to_quat(y_axis, slope)
    wall4_pos_z = -half_height + half_wall_height * np.cos(slope)
    wall4_pos_x = -half_length + half_thick + half_wall_height * np.sin(slope)
    wall4_pos = [wall4_pos_x, 0, wall4_pos_z]
    wall4_size = [half_thick, (half_width - wall_thickness)*length_scale,
                  half_wall_height]
    wall4_pose = sapien.Pose(wall4_pos, wall4_rot)

    if visual:
        builder.add_box_visual(half_size=wall1_size, pose=wall1_pose, material=visual_material)
        builder.add_box_visual(half_size=wall2_size, pose=wall2_pose, material=visual_material)
        builder.add_box_visual(half_size=wall3_size, pose=wall3_pose, material=visual_material)
        builder.add_box_visual(half_size=wall4_size, pose=wall4_pose, material=visual_material)
    if collision:
        builder.add_box_collision(half_size=wall1_size, pose=wall1_pose)
        builder.add_box_collision(half_size=wall2_size, pose=wall2_pose)
        builder.add_box_collision(half_size=wall3_size, pose=wall3_pose)
        builder.add_box_collision(half_size=wall4_size, pose=wall4_pose)

    # Set the initial pose
    builder.initial_pose = sapien.Pose(pos, quat)

    # Build the actor
    actor = builder.build_static(name=name)
    return actor


class Hobot2BridgeEnv(BaseBridgeEnv):
    SUPPORTED_OBS_MODES = ["rgb+depth+segmentation"]
    scene_setting: Literal["hobot2_flat_table"] = "hobot2_flat_table"

    # Note that the robot is positioned at
    # sapien.Pose(p=[0.127, 0.060, 0.85], q=[0, 0, 0, 1])
    # in the global frame.
    robot_pose_in_global = np.array([0.127, 0.060, 0.85])

    def __init__(self, add_tray: bool = True, **kwargs):
        kwargs["robot_cls"] = Hobot2BridgeFlatTable
        self._add_tray = add_tray
        super().__init__(**kwargs)

    # In hobot2, we use 10Hz control frequency
    # We'll also use a sim timestep of 5ms
    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=200, control_freq=10, spacing=20)

    def _load_scene(self, options: dict):
        super()._load_scene(options)

        if not self._add_tray:
            return

        # This center position was computed from Hobot
        box_pos = np.array([-0.23, -0.06, 0.88759])

        # Add the xy offset given the robot's position
        box_pos[:2] += self.robot_pose_in_global[:2]

        # Specify the box's size. Computed from the place algorithm's
        # sampling distribution.
        # TODO: Don't hardcode this. Add an alf config
        box_size = np.array([0.14, 0.24, 0.04])
        # Add a tolerance to account for the cube's size.
        tolerance = 0.05
        box_size[:2] += tolerance
        name = "hobot2_boundary"
        boundary = create_hobot2_boundary(self.scene, pos=box_pos,
                               outer_size=box_size,
                               visual=True,
                               wall_thickness=0.002,
                               slope=np.deg2rad(-60),
                               name=name)

        self.remove_object_from_greenscreen(boundary)
