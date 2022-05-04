from typing import List
from minerl.herobraine.env_specs.simple_embodiment import SIMPLE_KEYBOARD_ACTION, SimpleEmbodimentEnvSpec
from minerl.herobraine.hero.handler import Handler
import minerl.herobraine.hero.handlers as handlers
from minerl.herobraine.hero.mc import INVERSE_KEYMAP


SIMPLE_KEYBOARD_ACTIONS = [
    'forward',
    'jump'
]
TREECHOP_WORLD_GENERATOR_OPTIONS = '''{
    "coordinateScale": 684.412,
    "heightScale": 684.412,
    "lowerLimitScale": 512.0,
    "upperLimitScale": 512.0,
    "depthNoiseScaleX": 200.0,
    "depthNoiseScaleZ": 200.0,
    "depthNoiseScaleExponent": 0.5,
    "mainNoiseScaleX": 80.0,
    "mainNoiseScaleY": 160.0,
    "mainNoiseScaleZ": 80.0,
    "baseSize": 8.5,
    "stretchY": 12.0,
    "biomeDepthWeight": 1.0,
    "biomeDepthOffset": 0.0,
    "biomeScaleWeight": 1.0,
    "biomeScaleOffset": 0.0,
    "seaLevel": 1,
    "useCaves": false,
    "useDungeons": false,
    "dungeonChance": 8,
    "useStrongholds": false,
    "useVillages": false,
    "useMineShafts": false,
    "useTemples": false,
    "useMonuments": false,
    "useMansions": false,
    "useRavines": false,
    "useWaterLakes": false,
    "waterLakeChance": 4,
    "useLavaLakes": false,
    "lavaLakeChance": 80,
    "useLavaOceans": false,
    "fixedBiome": 4,
    "biomeSize": 4,
    "riverSize": 1,
    "dirtSize": 33,
    "dirtCount": 10,
    "dirtMinHeight": 0,
    "dirtMaxHeight": 256,
    "gravelSize": 33,
    "gravelCount": 8,
    "gravelMinHeight": 0,
    "gravelMaxHeight": 256,
    "graniteSize": 33,
    "graniteCount": 10,
    "graniteMinHeight": 0,
    "graniteMaxHeight": 80,
    "dioriteSize": 33,
    "dioriteCount": 10,
    "dioriteMinHeight": 0,
    "dioriteMaxHeight": 80,
    "andesiteSize": 33,
    "andesiteCount": 10,
    "andesiteMinHeight": 0,
    "andesiteMaxHeight": 80,
    "coalSize": 17,
    "coalCount": 20,
    "coalMinHeight": 0,
    "coalMaxHeight": 128,
    "ironSize": 9,
    "ironCount": 20,
    "ironMinHeight": 0,
    "ironMaxHeight": 64,
    "goldSize": 9,
    "goldCount": 2,
    "goldMinHeight": 0,
    "goldMaxHeight": 32,
    "redstoneSize": 8,
    "redstoneCount": 8,
    "redstoneMinHeight": 0,
    "redstoneMaxHeight": 16,
    "diamondSize": 8,
    "diamondCount": 1,
    "diamondMinHeight": 0,
    "diamondMaxHeight": 16,
    "lapisSize": 7,
    "lapisCount": 1,
    "lapisCenterHeight": 16,
    "lapisSpread": 16
}'''


class SimpleExplore(SimpleEmbodimentEnvSpec):
    def __init__(self, forest_only, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'SimpleExplore-v0'
        self.forest_only = forest_only
        super().__init__(*args, **kwargs)

    def create_rewardables(self) -> List[Handler]:
        return []

    def create_agent_start(self) -> List[Handler]:
        return []

    def create_agent_handlers(self) -> List[Handler]:
        return []

    def create_server_world_generators(self) -> List[Handler]:
        gen_options = TREECHOP_WORLD_GENERATOR_OPTIONS if self.forest_only else None
        return [
            handlers.DefaultWorldGenerator(force_reset="true", generator_options=gen_options)
        ]
        

    def create_actionables(self) -> List[Handler]:
        return [
            handlers.KeybasedCommandAction(k, v) for k, v in INVERSE_KEYMAP.items()
            if k in SIMPLE_KEYBOARD_ACTION
        ] + [handlers.CameraAction()]

    def create_server_quit_producers(self) -> List[Handler]:
        return []

    def create_server_decorators(self) -> List[Handler]:
        return []

    def create_server_initial_conditions(self) -> List[Handler]:
        return [
            handlers.TimeInitialCondition(
                allow_passage_of_time=False
            ),
            handlers.SpawningInitialCondition(
                allow_spawning=True
            )
        ]

    def determine_success_from_rewards(self, rewards: list) -> bool:
        return False

    def is_from_folder(self, folder: str) -> bool:
        return False

    def get_docstring(self):
        return """"""
