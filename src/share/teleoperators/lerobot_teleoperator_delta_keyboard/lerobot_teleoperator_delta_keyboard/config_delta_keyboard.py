#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.teleoperators import TeleoperatorConfig


@dataclass
class KeyboardAxisBinding:
    """
    Mapping for one velocity axis.

    Example:
        x: pos_key='d', neg_key='a', scale=0.1
    """

    pos_key: str | None = None
    neg_key: str | None = None
    scale: float = 1.0
    enabled: bool = True


@dataclass
class KeyboardEventBinding:
    """
    Mapping for a teleop event.

    Attributes:
        key: key token, e.g. 'space', 'p', 'enter'
        toggle: if True, event toggles on rising edge. Otherwise it is level-triggered.
    """

    key: str
    toggle: bool = False


@TeleoperatorConfig.register_subclass("keyboard_velocity")
@dataclass
class KeyboardVelocityTeleopConfig(TeleoperatorConfig):
    """
    Configurable keyboard teleoperator that outputs per-axis velocities.

    Key tokens are normalized strings, for example:
        - Characters: 'w', 'a', 's', 'd', 'j', 'k'
        - Special keys: 'up', 'down', 'left', 'right', 'space', 'shift', 'shift_r',
          'ctrl_l', 'ctrl_r', 'esc', 'enter', 'tab', 'backspace'
    """

    x: KeyboardAxisBinding = field(
        default_factory=lambda: KeyboardAxisBinding(pos_key="d", neg_key="a", scale=0.1)
    )
    y: KeyboardAxisBinding = field(
        default_factory=lambda: KeyboardAxisBinding(pos_key="right", neg_key="left", scale=0.1)
    )
    z: KeyboardAxisBinding = field(
        default_factory=lambda: KeyboardAxisBinding(pos_key="up", neg_key="down", scale=0.1)
    )
    rx: KeyboardAxisBinding = field(
        default_factory=lambda: KeyboardAxisBinding(pos_key="o", neg_key="u", scale=0.5)
    )
    ry: KeyboardAxisBinding = field(
        default_factory=lambda: KeyboardAxisBinding(pos_key="j", neg_key="l", scale=0.5)
    )
    rz: KeyboardAxisBinding = field(
        default_factory=lambda: KeyboardAxisBinding(pos_key="w", neg_key="s", scale=0.5)
    )

    event_bindings: dict[str, KeyboardEventBinding] = field(default_factory=dict)

    include_intervention_event: bool = True
    escape_disconnects: bool = True