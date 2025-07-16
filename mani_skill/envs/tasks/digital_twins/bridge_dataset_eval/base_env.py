"""
Base environment for Bridge dataset environments
"""
import os
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import sapien
import torch
from sapien.physx import PhysxMaterial
from transforms3d.quaternions import quat2mat

from mani_skill import ASSET_DIR
from mani_skill.agents.controllers.pd_ee_pose import PDEEPoseControllerConfig
from mani_skill.agents.controllers.pd_joint_pos import PDJointPosMimicControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.widowx.widowx import WidowX250S
from mani_skill.envs.tasks.digital_twins.base_env import BaseDigitalTwinEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, io_utils, sapien_utils
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig

from hobot2 import HOBOT2_ROOT

BRIDGE_DATASET_ASSET_PATH = ASSET_DIR / "tasks/bridge_v2_real2sim_dataset/"
HOBOT2_ASSET_PATH = ASSET_DIR / "hobot2_mani_skill_assets"
# Real2Sim tuned WidowX250S robot
@register_agent(asset_download_ids=["widowx250s"])
class WidowX250SBridgeDatasetFlatTable(WidowX250S):
    uid = "widowx250s_bridgedataset_flat_table"
    arm_joint_names = [
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
    ]
    gripper_joint_names = ["left_finger", "right_finger"]

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="3rd_view_camera",  # the camera used in the Bridge dataset
                pose=sapien.Pose(
                    [0.00, -0.16, 0.36],
                    [0.8992917, -0.09263245, 0.35892478, 0.23209205],
                ),
                width=640,
                height=480,
                entity_uid="base_link",
                intrinsic=np.array(
                    [[623.588, 0, 319.501], [0, 623.588, 239.545], [0, 0, 1]]
                ),  # logitech C920
            ),
        ]

    arm_stiffness = [
        1169.7891719504198,
        730.0,
        808.4601346394447,
        1229.1299089624076,
        1272.2760456418862,
        1056.3326605132252,
    ]
    arm_damping = [
        330.0,
        180.0,
        152.12036565582588,
        309.6215302722146,
        201.04998711007383,
        269.51458932695414,
    ]

    arm_force_limit = [200, 200, 100, 100, 100, 100]
    arm_friction = 0.0
    arm_vel_limit = 1.5
    arm_acc_limit = 2.0

    gripper_stiffness = 1000
    gripper_damping = 200
    gripper_pid_stiffness = 1000
    gripper_pid_damping = 200
    gripper_pid_integral = 300
    gripper_force_limit = 60
    gripper_vel_limit = 0.12
    gripper_acc_limit = 0.50
    gripper_jerk_limit = 5.0

    @property
    def _controller_configs(self):
        arm_common_kwargs = dict(
            joint_names=self.arm_joint_names,
            pos_lower=-1.0,  # dummy limit, which is unused since normalize_action=False
            pos_upper=1.0,
            rot_lower=-np.pi / 2,
            rot_upper=np.pi / 2,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link="ee_gripper_link",
            urdf_path=self.urdf_path,
            normalize_action=False,
        )
        arm_pd_ee_target_delta_pose_align2 = PDEEPoseControllerConfig(
            **arm_common_kwargs, use_target=True
        )

        extra_gripper_clearance = 0.001  # since real gripper is PID, we use extra clearance to mitigate PD small errors; also a trick to have force when grasping
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            joint_names=self.gripper_joint_names,
            lower=0.015 - extra_gripper_clearance,
            upper=0.037 + extra_gripper_clearance,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            normalize_action=True,
            drive_mode="force",
        )
        controller = dict(
            arm=arm_pd_ee_target_delta_pose_align2, gripper=gripper_pd_joint_pos
        )
        return dict(arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos=controller)


# Tuned for the sink setup
@register_agent(asset_download_ids=["widowx250s"])
class WidowX250SBridgeDatasetSink(WidowX250SBridgeDatasetFlatTable):
    uid = "widowx250s_bridgedataset_sink"

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="3rd_view_camera",  # the camera used for real evaluation for the sink setup
                pose=sapien.Pose(
                    [-0.00300001, -0.21, 0.39], [-0.907313, 0.0782, -0.36434, -0.194741]
                ),
                entity_uid="base_link",
                width=640,
                # fov=1.5,
                height=480,
                near=0.01,
                far=10,
                intrinsic=np.array(
                    [[623.588, 0, 319.501], [0, 623.588, 239.545], [0, 0, 1]]
                ),
            )
        ]


class BaseBridgeEnv(BaseDigitalTwinEnv):
    """Base Digital Twin environment for digital twins of the BridgeData v2"""

    MODEL_JSON = "info_bridge_custom_v0.json"
    SUPPORTED_OBS_MODES = ["rgb+segmentation"]
    SUPPORTED_REWARD_MODES = ["none"]
    scene_setting: Literal["flat_table", "sink"] = "flat_table"
    objects_excluded_from_greenscreening: List[str] = []
    """object ids that should not be greenscreened"""

    obj_static_friction = 0.5
    obj_dynamic_friction = 0.5

    HOBOT2_CUSTOM_FILES = (Path(HOBOT2_ROOT) /
               "environment/hobot2_mani_skill_files")

    def __init__(
        self,
        obj_names: List[str],
        xyz_configs: torch.Tensor,
        quat_configs: torch.Tensor,
        obj_perturb_radius: float = 0.0,
        obj_sample_orientations: bool = False,
        **kwargs,
    ):
        self.objs: Dict[str, Actor] = dict()
        self.obj_names = obj_names
        self.xyz_configs = xyz_configs
        self.quat_configs = quat_configs
        self.obj_perturb_radius = obj_perturb_radius
        self.obj_sample_orientations = obj_sample_orientations
        if (
            self.scene_setting == "flat_table"
            or self.scene_setting == "hobot2_flat_table"
        ):
            self.rgb_overlay_paths = {
                "3rd_view_camera": str(
                    BRIDGE_DATASET_ASSET_PATH / "real_inpainting/bridge_real_eval_1.png"
                )
            }
            robot_cls = WidowX250SBridgeDatasetFlatTable
        elif self.scene_setting == "sink":
            self.rgb_overlay_paths = {
                "3rd_view_camera": str(
                    BRIDGE_DATASET_ASSET_PATH / "real_inpainting/bridge_sink.png"
                )
            }
            robot_cls = WidowX250SBridgeDatasetSink

        if kwargs.get("robot_cls", None) is not None:
            robot_cls = kwargs["robot_cls"]
            del kwargs["robot_cls"]

        if "hobot2" in self.scene_setting:
            self.model_db = io_utils.load_json(self.HOBOT2_CUSTOM_FILES /
                                               self.MODEL_JSON)
        else:
            self.model_db: Dict[str, Dict] = io_utils.load_json(
                BRIDGE_DATASET_ASSET_PATH / "custom/" / self.MODEL_JSON
            )

        super().__init__(
            robot_uids=robot_cls,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=500, control_freq=5, spacing=20)

    @property
    def _default_human_render_camera_configs(self):
        sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera",
            pose=sapien.Pose(
                [0.00, -0.16, 0.336], [0.909182, -0.0819809, 0.347277, 0.214629]
            ),
            width=512,
            height=512,
            intrinsic=np.array(
                [[623.588, 0, 319.501], [0, 623.588, 239.545], [0, 0, 1]]
            ),
            near=0.01,
            far=100,
            mount=self.agent.robot.links_map["base_link"],
        )

    def _build_actor_helper(
        self,
        model_id: str,
        scale: float = 1,
        kinematic: bool = False,
        initial_pose: Pose = None,
    ):
        """helper function to build actors by ID directly and auto configure physical materials"""
        density = self.model_db[model_id].get("density", 1000)
        physical_material = PhysxMaterial(
            static_friction=self.obj_static_friction,
            dynamic_friction=self.obj_dynamic_friction,
            restitution=0.0,
        )
        builder = self.scene.create_actor_builder()

        # Note: We won't perform mesh decomposition for the ManiSkill bridge
        # dataset objects. This is the default behavior.
        mesh_decomposition = "none"
        model_dir = BRIDGE_DATASET_ASSET_PATH / "custom" / "models" / model_id

        # Reroute to Hobot2 assets if necessary and enable mesh decomposition
        # for custom objects. This is necessary for objects like bowls to
        # have proper convex shapes instead of a simple bounding box.
        if not model_dir.exists():
            model_dir = HOBOT2_ASSET_PATH / model_id
            mesh_decomposition = "coacd"

        collision_file = str(model_dir / "collision.obj")
        builder.add_multiple_convex_collisions_from_file(
            filename=collision_file,
            scale=[scale] * 3,
            material=physical_material,
            density=density,
            decomposition=mesh_decomposition
        )

        visual_file = str(model_dir / "textured.obj")
        if not os.path.exists(visual_file):
            visual_file = str(model_dir / "textured.dae")
            if not os.path.exists(visual_file):
                visual_file = str(model_dir / "textured.glb")
        builder.add_visual_from_file(filename=visual_file, scale=[scale] * 3)
        if initial_pose is not None:
            builder.initial_pose = initial_pose
        if kinematic:
            actor = builder.build_kinematic(name=model_id)
        else:
            actor = builder.build(name=model_id)
        return actor

    def _load_lighting(self, options: dict):
        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light(
            [0, 0, -1],
            [2.2, 2.2, 2.2],
            shadow=False,
            shadow_scale=5,
            shadow_map_size=2048,
        )
        self.scene.add_directional_light([-1, -0.5, -1], [0.7, 0.7, 0.7])
        self.scene.add_directional_light([1, 1, -1], [0.7, 0.7, 0.7])

    def _load_agent(self, options: dict):
        super()._load_agent(
            options, sapien.Pose(p=[0.127, 0.060, 0.85], q=[0, 0, 0, 1])
        )

    def _load_scene(self, options: dict):
        # original SIMPLER envs always do this? except for open drawer task
        for i in range(self.num_envs):
            sapien_utils.set_articulation_render_material(
                self.agent.robot._objs[i], specular=0.9, roughness=0.3
            )

        # load background
        builder = self.scene.create_actor_builder()
        scene_pose = sapien.Pose(q=[0.707, 0.707, 0, 0])
        scene_offset = np.array([-2.0634, -2.8313, 0.0])
        if self.scene_setting == "hobot2_flat_table":
            scene_file = str(self.HOBOT2_CUSTOM_FILES /
                "hobot2_bridge_tabletop_scene.glb"
            )

        elif self.scene_setting == "flat_table":
            scene_file = str(BRIDGE_DATASET_ASSET_PATH / "stages/bridge_table_1_v1.glb")

        elif self.scene_setting == "sink":
            scene_file = str(BRIDGE_DATASET_ASSET_PATH / "stages/bridge_table_1_v2.glb")
        builder.add_nonconvex_collision_from_file(scene_file, pose=scene_pose)
        builder.add_visual_from_file(scene_file, pose=scene_pose)

        builder.initial_pose = sapien.Pose(-scene_offset)
        builder.build_static(name="arena")

        # TODO: Can add scale sampling here if needed
        for name in self.obj_names:
            scale = self.model_db[name].get("scale", 1.0)
            self.objs[name] = self._build_actor_helper(name, scale=scale)

        self.xyz_configs = common.to_tensor(self.xyz_configs, device=self.device).to(
            torch.float32
        )
        self.quat_configs = common.to_tensor(self.quat_configs, device=self.device).to(
            torch.float32
        )

        if self.scene_setting == "sink":
            self.sink = self._build_actor_helper(
                "sink",
                kinematic=True,
                initial_pose=sapien.Pose([-0.16, 0.13, 0.88], [1, 0, 0, 0]),
            )

        for obj_name in self.objects_excluded_from_greenscreening:
            self.remove_object_from_greenscreen(self.objs[obj_name])
        self.remove_object_from_greenscreen(self.agent.robot)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # NOTE: this part of code is not GPU parallelized
        with torch.device(self.device):
            reset_objects = options.get("reset_objects", True)
            b = len(env_idx)
            if "episode_id" in options:
                if isinstance(options["episode_id"], int):
                    options["episode_id"] = torch.tensor([options["episode_id"]])
                    assert len(options["episode_id"]) == b
                pos_episode_ids = (
                    options["episode_id"]
                    % (len(self.xyz_configs) * len(self.quat_configs))
                ) // len(self.quat_configs)
                quat_episode_ids = options["episode_id"] % len(self.quat_configs)
            else:
                pos_episode_ids = torch.randint(0, len(self.xyz_configs), size=(b,))
                quat_episode_ids = torch.randint(0, len(self.quat_configs), size=(b,))

            def sample_random_2d_displacements(n: int, r: float):
                u = torch.rand(n)  # Uniform[0, 1)
                radii = r * torch.sqrt(u)  # sqrt ensures uniform density
                angles = torch.rand(n) * 2 * torch.pi  # Uniform[0, 2pi)
                displacements = torch.stack(
                    (radii * torch.cos(angles), radii * torch.sin(angles)),
                    dim=1
                )
                return displacements

            def sample_random_yaw_quaternions(n: int):
                angles = torch.rand(n) * 2 * torch.pi  # Uniform[0, 2pi)
                half_angles = angles / 2

                w = torch.cos(half_angles)
                x = torch.zeros(n)
                y = torch.zeros(n)
                z = torch.sin(half_angles)

                quaternions = torch.stack((w, x, y, z), dim=1)  # shape (n, 4)
                return quaternions

            if reset_objects:
                for i, actor in enumerate(self.objs.values()):
                    xyz = self.xyz_configs[pos_episode_ids, i]
                    quat = self.quat_configs[quat_episode_ids, i]

                    bs = xyz.shape[0]
                    if self.obj_perturb_radius > 0.0:
                        xyz[:, :2] += sample_random_2d_displacements(
                            bs, self.obj_perturb_radius
                        )

                    if self.obj_sample_orientations:
                        quat = sample_random_yaw_quaternions(bs)

                    actor.set_pose(
                        Pose.create_from_pq(p=xyz,
                                            q=quat)
                    )

            # measured values for bridge dataset
            if self.scene_setting == "flat_table" or self.scene_setting == "hobot2_flat_table":
                qpos = np.array(
                    [
                        -0.01840777,
                        0.0398835,
                        0.22242722,
                        -0.00460194,
                        1.36524296,
                        0.00153398,
                        0.037,
                        0.037,
                    ]
                )

                self.agent.robot.set_pose(
                    sapien.Pose([0.147, 0.028, 0.870], q=[0, 0, 0, 1])
                )
            elif self.scene_setting == "sink":
                qpos = np.array(
                    [
                        -0.2600599,
                        -0.12875618,
                        0.04461369,
                        -0.00652761,
                        1.7033415,
                        -0.26983038,
                        0.037,
                        0.037,
                    ]
                )
                self.agent.robot.set_pose(
                    sapien.Pose([0.127, 0.060, 0.85], q=[0, 0, 0, 1])
                )
            self.agent.reset(init_qpos=qpos)

    def _evaluate(
        self,
        success_require_src_completely_on_target=True,
        z_flag_required_offset=0.02,
        **kwargs,
    ):
        return {}

    def is_final_subtask(self):
        # whether the current subtask is the final one, only meaningful for long-horizon tasks
        return True
