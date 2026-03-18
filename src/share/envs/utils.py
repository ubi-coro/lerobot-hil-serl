from __future__ import annotations

import types
from typing import TYPE_CHECKING, get_origin, Union, get_args

if TYPE_CHECKING:
    from lerobot.teleoperators import Teleoperator
    from lerobot.robots import Robot


def check_task_frame_robot(robot_dict: dict[str, "Robot"]):
    is_task_frame_robot = {}
    for name, r in robot_dict.items():
        is_task_frame_robot[name] = hasattr(r, "set_task_frame")

    return is_task_frame_robot


def check_delta_teleoperator(teleop_dict: dict[str, "Teleoperator"]):
    is_delta_teleoperator = {}
    for name, t in teleop_dict.items():
        is_delta_teleoperator[name] = all(ft.startswith("delta_") for ft in t.action_features.keys())

    return is_delta_teleoperator


def is_union_with_dict(field_type) -> bool:
    origin = get_origin(field_type)
    if origin is types.UnionType or origin is Union:
        return any(get_origin(arg) is dict for arg in get_args(field_type))
    return False
