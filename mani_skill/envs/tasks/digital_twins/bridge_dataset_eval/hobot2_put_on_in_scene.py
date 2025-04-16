import torch

from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval.hobot2_base_env import (
    Hobot2BridgeEnv,
)
from mani_skill.utils.registration import register_env


@register_env(
    "Hobot2RedCubeScene",
    max_episode_steps=None,
    asset_download_ids=["bridge_v2_real2sim"],
)
class Hobot2RedCubeScene(Hobot2BridgeEnv):
    scene_setting = "hobot2_flat_table"
    MODEL_JSON = "hobot2_object_db.json"
    objects_excluded_from_greenscreening = [
        "baked_red_cube_3cm",
        "baked_yellow_cube_3cm",
    ]

    def __init__(
        self,
        **kwargs,
    ):
        # We don't care about the yellow cube, so just teleport it away for now

        xyz_configs = torch.tensor([[-0.125, 0, 0.887529], [100, 0, 0]])
        quat_configs = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]])
        source_obj_name = "baked_red_cube_3cm"
        target_obj_name = "baked_yellow_cube_3cm"
        super().__init__(
            obj_names=[source_obj_name, target_obj_name],
            xyz_configs=xyz_configs.unsqueeze(0),
            quat_configs=quat_configs.unsqueeze(0),
            **kwargs,
        )


@register_env(
    "Hobot2GreenYellowCubeScene",
    max_episode_steps=None,
    asset_download_ids=["bridge_v2_real2sim"],
)
class Hobot2GreenYellowCubeScene(Hobot2BridgeEnv):
    scene_setting = "hobot2_flat_table"
    MODEL_JSON = "hobot2_object_db.json"
    objects_excluded_from_greenscreening = [
        "baked_green_cube_3cm",
        "baked_yellow_cube_3cm",
    ]

    def __init__(
            self,
            **kwargs,
    ):
        # We don't care about the yellow cube, so just teleport it away for now

        xyz_configs = torch.tensor([[-0.175, 0, 0.887529], [-0.075, 0,
                                                           0.887529]])
        quat_configs = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]])
        source_obj_name = "baked_green_cube_3cm"
        target_obj_name = "baked_yellow_cube_3cm"
        super().__init__(
            add_tray=False,
            obj_names=[source_obj_name, target_obj_name],
            xyz_configs=xyz_configs.unsqueeze(0),
            quat_configs=quat_configs.unsqueeze(0),
            **kwargs,
        )


@register_env(
    "Hobot2SpoonTowelScene",
    max_episode_steps=None,
    asset_download_ids=["bridge_v2_real2sim"],
)
class Hobot2SpoonTowelScene(Hobot2BridgeEnv):
    scene_setting = "hobot2_flat_table"
    MODEL_JSON = "hobot2_object_db.json"
    objects_excluded_from_greenscreening = [
        "bridge_spoon_generated_modified",
        "table_cloth_generated_shorter",
    ]

    def __init__(
            self,
            **kwargs,
    ):
        # We don't care about the yellow cube, so just teleport it away for now

        # xyz_configs = torch.tensor([[-0.20, 0.05, 0.887529], [-0.075, 0.05,
        #                                                     0.887529]])
        # xyz_configs = torch.tensor([[-0.25, 0.05, 0.887529], [-0.075, 0.0,
        #                                                       0.887529]])
        xyz_configs = torch.tensor([[-0.125, 0.10, 0.887529], [-0.075, -0.10,
                                                              0.887529]])
        # quat_configs = torch.tensor([[0.707, 0.0, 0, 0.707], [1, 0, 0, 0]])
        quat_configs = torch.tensor([[1.0, 0.0, 0, 0.0], [1, 0, 0, 0]])
        super().__init__(
            add_tray=True,
            obj_names=self.objects_excluded_from_greenscreening,
            xyz_configs=xyz_configs.unsqueeze(0),
            quat_configs=quat_configs.unsqueeze(0),
            **kwargs,
        )


@register_env(
    "Hobot2RedCubeTowelPlateScene",
    max_episode_steps=None,
    asset_download_ids=["bridge_v2_real2sim"],
)
class Hobot2RedCubeTowelPlateScene(Hobot2BridgeEnv):
    scene_setting = "hobot2_flat_table"
    MODEL_JSON = "hobot2_object_db.json"
    objects_excluded_from_greenscreening = [
        "bridge_spoon_generated_modified",
        # "baked_red_cube_3cm",
        "table_cloth_generated_shorter",
        "bridge_plate_objaverse",
    ]

    def __init__(
            self,
            **kwargs,
    ):
        xyz_configs = torch.tensor([[-0.20, 0, 0.887529],
                                    [-0.075, -0.10, 0.88759],
                                    [-0.075, 0.10, 0.88759]])
        quat_configs = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
        super().__init__(
            add_tray=False,
            obj_names=self.objects_excluded_from_greenscreening,
            xyz_configs=xyz_configs[:3].unsqueeze(0),
            quat_configs=quat_configs[:3].unsqueeze(0),
            **kwargs,
        )
