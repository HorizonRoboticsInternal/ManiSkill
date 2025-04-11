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
    "Hobot2RedCubeTowelPlateScene",
    max_episode_steps=None,
    asset_download_ids=["bridge_v2_real2sim"],
)
class Hobot2RedCubeTowelPlateScene(Hobot2BridgeEnv):
    scene_setting = "hobot2_flat_table"
    MODEL_JSON = "hobot2_object_db.json"
    objects_excluded_from_greenscreening = [
        "baked_red_cube_3cm",
        "bridge_plate_objaverse",
        "table_cloth_generated_shorter"
    ]

    def __init__(
            self,
            **kwargs,
    ):
        xyz_configs = torch.tensor([[-0.05, 0, 0.887529],
                                    [-0.15, -0.10, 0.88759],
                                    [-0.15, 0.10, 0.88759]])
        quat_configs = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
        super().__init__(
            obj_names=self.objects_excluded_from_greenscreening,
            xyz_configs=xyz_configs.unsqueeze(0),
            quat_configs=quat_configs.unsqueeze(0),
            **kwargs,
        )
