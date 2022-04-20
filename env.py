from typing import List
from minerl.herobraine.env_specs.simple_embodiment import SIMPLE_KEYBOARD_ACTION, SimpleEmbodimentEnvSpec
from minerl.herobraine.hero.handler import Handler
import minerl.herobraine.hero.handlers as handlers
from minerl.herobraine.hero.mc import INVERSE_KEYMAP


SIMPLE_KEYBOARD_ACTIONS = [
    'forward',
    'jump'
]


class SimpleExplore(SimpleEmbodimentEnvSpec):
    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'SimpleExplore-v0'
        super().__init__(*args, **kwargs)

    def create_rewardables(self) -> List[Handler]:
        return []

    def create_agent_start(self) -> List[Handler]:
        return []

    def create_agent_handlers(self) -> List[Handler]:
        return []

    def create_server_world_generators(self) -> List[Handler]:
        return [
            handlers.DefaultWorldGenerator(force_reset="true")
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
