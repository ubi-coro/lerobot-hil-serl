from typing import Any

from lerobot.cameras import Camera
from lerobot.envs import RobotEnv
from lerobot.envs.configs import HILSerlProcessorConfig
from lerobot.robots import Robot
from lerobot.sim.mujoco_utils.sim_singleton import SimManager


class SimRobotEnv(RobotEnv):
    """Gym environment for simulation control with human intervention support."""

    def __init__(
        self,
        robot_dict: dict[str, Robot],
        cameras: dict[str, Camera] | None = None,
        processor: HILSerlProcessorConfig | None = None
    ) -> None:
        """Initialize simulation environment with configuration options.

        Args:
            robot: Robot interface for hardware communication.
            use_gripper: Whether to include gripper in action space.
            display_cameras: Whether to show camera feeds during execution.
            reset_pose: Joint positions for environment reset.
            reset_time_s: Time to wait during reset.
        """
        self.sim = SimManager.get()
        super().__init__(robot_dict, cameras, processor)

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        _ = self.sim.reset(seed=seed, options=options)
        obs, events = super().reset(seed=seed, options=options)
        return obs, events

    def _send_actions(self, action):
        """Execute one environment step with given action."""
        super()._send_actions(action)
        self.sim.step()

    def _setup_spaces(self) -> None:
        self.sim.action_order = self._joint_names_list # TODO(jzilke): move this to SimSession init
        self.sim.reset()
        super()._setup_spaces()
