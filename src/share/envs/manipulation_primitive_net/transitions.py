from dataclasses import dataclass

from draccus import ChoiceRegistry

@dataclass
class MP_Transition(ChoiceRegistry):
    def check(self, obs: dict, info: dict) -> bool:
        raise NotImplementedError
