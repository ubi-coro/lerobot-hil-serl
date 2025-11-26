"""Experiment mapping bridge between GUI selections and experiments package."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# Ensure pynput uses a headless backend when no DISPLAY is available so
# experiments can be imported inside the FastAPI worker.
if not os.environ.get("DISPLAY"):
	os.environ.setdefault("PYNPUT_BACKEND", "dummy")

try:  # Python 3.11+
	import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for older interpreters
	import tomli as tomllib  # type: ignore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentMapping:
	"""Describes how a GUI mode maps to an experiment configuration."""

	operation_mode: str
	demo_mode: bool
	experiment_type: str
	description: str
	supports_policy: bool = False
	default_policy_path: Optional[str] = None

	def with_policy_override(self, policy_path: Optional[str]) -> "ExperimentMapping":
		"""Return a copy with the default policy path replaced if provided."""

		if not policy_path or not self.supports_policy:
			return self
		return replace(self, default_policy_path=policy_path)


class ExperimentConfigMapper:
	"""Maps GUI selections to experiment configs defined under ``src/experiments``."""

	_CONFIG_PATH = Path(__file__).with_name("hardware_profiles.toml")
	_MODE_NORMALIZATION: Dict[str, str] = {
		"bimanual": "bimanual",
		"dual": "bimanual",
		"both": "bimanual",
		"bi_viperx": "bimanual",
		"left": "left",
		"left_only": "left",
		"left_arm": "left",
		"single_left": "left",
		"aloha_left": "left",
		"right": "right",
		"right_only": "right",
		"right_arm": "right",
		"single_right": "right",
		"aloha_right": "right",
	}

	@classmethod
	@lru_cache(maxsize=1)
	def _load_profile_data(cls) -> Dict[str, Any]:
		if not cls._CONFIG_PATH.exists():
			raise FileNotFoundError(f"hardware profile file not found: {cls._CONFIG_PATH}")
		with cls._CONFIG_PATH.open("rb") as fp:
			data = tomllib.load(fp)
		return data

	@classmethod
	def normalize_operation_mode(cls, mode: Optional[str]) -> str:
		if not mode:
			return "bimanual"
		key = mode.lower().strip()
		return cls._MODE_NORMALIZATION.get(key, "bimanual")

	@classmethod
	def _lookup_experiment(cls, operation_mode: str, demo_mode: bool) -> Tuple[str, bool]:
		data = cls._load_profile_data()
		gui_map: Dict[str, Dict[str, str]] = data.get("gui_map", {})
		teleop_map = gui_map.get("teleop", {})
		demo_map = gui_map.get("demo", {})

		if demo_mode:
			experiment = demo_map.get(operation_mode)
			if experiment:
				return experiment, True
			logger.warning(
				"No demo mapping for mode '%s'; falling back to teleop experiment without policy support",
				operation_mode,
			)
			experiment = teleop_map.get(operation_mode)
			if not experiment:
				available = sorted({**teleop_map, **demo_map})
				raise ValueError(
					f"No experiment mapping for operation_mode='{operation_mode}' (demo). Available: {available}"
				)
			return experiment, False

		experiment = teleop_map.get(operation_mode)
		if not experiment:
			available = sorted(teleop_map)
			raise ValueError(
				f"No experiment mapping for operation_mode='{operation_mode}'. Available: {available}"
			)
		return experiment, False

	@classmethod
	def resolve_mapping(
		cls,
		operation_mode: Optional[str],
		demo_mode: bool = False,
		policy_path_override: Optional[str] = None,
	) -> ExperimentMapping:
		"""Resolve GUI selection into :class:`ExperimentMapping`."""

		normalized = cls.normalize_operation_mode(operation_mode)
		experiment_type, supports_policy = cls._lookup_experiment(normalized, demo_mode)

		cls._ensure_experiments_registered()
		from lerobot.envs.configs import EnvConfig

		try:
			env_class = EnvConfig.get_choice_class(experiment_type)
		except KeyError as exc:  # pragma: no cover - validation guard
			raise ValueError(f"Experiment '{experiment_type}' is not registered: {exc}") from exc

		description = (env_class.__doc__ or experiment_type).strip().split("\n")[0].strip()

		policy_defaults: Dict[str, str] = (
			cls._load_profile_data().get("gui_defaults", {}).get("policy_paths", {})
		)
		default_policy_path = policy_defaults.get(experiment_type)

		mapping = ExperimentMapping(
			operation_mode=normalized,
			demo_mode=demo_mode and supports_policy,
			experiment_type=experiment_type,
			description=description,
			supports_policy=supports_policy,
			default_policy_path=default_policy_path,
		)
		return mapping.with_policy_override(policy_path_override)

	@classmethod
	def create_env_from_gui_selection(
		cls,
		operation_mode: str,
		demo_mode: bool = False,
		policy_path_override: Optional[str] = None,
		device: str = "cpu",
		use_cameras: bool = True,
	) -> Tuple[Any, Any, Any, Any, ExperimentMapping]:
		"""Instantiate an experiment-defined environment for the GUI."""

		mapping = cls.resolve_mapping(operation_mode, demo_mode, policy_path_override)

		from lerobot.envs.configs import EnvConfig

		cls._ensure_experiments_registered()
		env_class = EnvConfig.get_choice_class(mapping.experiment_type)
		env_cfg = env_class()

		if not use_cameras:
			try:
				env_cfg.cameras = {}
			except Exception:
				logger.debug("Unable to clear environment cameras", exc_info=True)
			# Clear per-robot embedded cameras if present.
			try:
				robots = env_cfg.robot if isinstance(env_cfg.robot, dict) else {"_": env_cfg.robot}
				for cfg in robots.values():
					if hasattr(cfg, "cameras"):
						cfg.cameras = {}
			except Exception:
				logger.debug("Unable to clear robot cameras", exc_info=True)

		# Keyboard listeners (pynput) are not available in headless server environments.
		# When we force the dummy backend, disable key mappings so the processor does
		# not try to spawn a Listener thread.
		if os.environ.get("PYNPUT_BACKEND") == "dummy":
			try:
				env_cfg.processor.events.key_mapping = {}
			except Exception:
				logger.debug("Failed to clear key mappings for headless mode", exc_info=True)

		# Use lazy camera connection: connect cameras only if requested
		env, env_processor, action_processor = env_cfg.make(device=device, connect_cameras=use_cameras)
		logger.info(
			"Created environment for mode=%s demo=%s using experiment=%s",
			mapping.operation_mode,
			mapping.demo_mode,
			mapping.experiment_type,
		)

		return env, env_processor, action_processor, env_cfg, mapping

	@classmethod
	def list_available_mappings(cls) -> Dict[str, Dict[str, Any]]:
		"""Return a dictionary summarising all GUI → experiment mappings."""

		data = {}
		for demo_mode in (False, True):
			for candidate in ("bimanual", "left", "right"):
				try:
					mapping = cls.resolve_mapping(candidate, demo_mode)
				except Exception:
					continue
				key = f"{candidate}_{'demo' if demo_mode else 'teleop'}"
				data[key] = {
					"operation_mode": mapping.operation_mode,
					"demo_mode": mapping.demo_mode,
					"experiment_type": mapping.experiment_type,
					"description": mapping.description,
					"supports_policy": mapping.supports_policy,
					"default_policy_path": mapping.default_policy_path,
				}
		return data

	@staticmethod
	@lru_cache(maxsize=1)
	def _ensure_experiments_registered() -> None:
		"""Import the experiments package once to register EnvConfig subclasses."""

		try:
			import experiments  # noqa: F401
		except Exception as exc:  # pragma: no cover - defensive guard
			logger.error("Failed to import experiments package: %s", exc)
			raise

	@classmethod
	def get_demo_defaults(cls, experiment_type: str) -> Optional[Dict[str, Any]]:
		"""Get pre-configured demo defaults for a given experiment type.
		
		These defaults are used to pre-fill the Demo/Replay form when the user
		connects with a demo-enabled experiment configuration.
		
		Returns None if no demo defaults are configured for this experiment.
		"""
		data = cls._load_profile_data()
		demo_defaults = data.get("demo_defaults", {})
		return demo_defaults.get(experiment_type)

	@classmethod
	def is_demo_experiment(cls, experiment_type: str) -> bool:
		"""Check if an experiment type is configured as a demo experiment."""
		data = cls._load_profile_data()
		demo_map = data.get("gui_map", {}).get("demo", {})
		return experiment_type in demo_map.values()

	@classmethod
	def get_demo_config_for_mode(cls, operation_mode: str) -> Optional[Dict[str, Any]]:
		"""Get the full demo configuration for an operation mode.
		
		Returns a dict with:
		- experiment_type: the experiment name
		- policy_path: pre-configured policy path
		- task_description: task description
		- fps, episode_time_s, reset_time_s, num_episodes: timing defaults
		- interactive: whether intervention is enabled by default
		
		Returns None if no demo is configured for this mode.
		"""
		normalized = cls.normalize_operation_mode(operation_mode)
		data = cls._load_profile_data()
		demo_map = data.get("gui_map", {}).get("demo", {})
		
		experiment_type = demo_map.get(normalized)
		if not experiment_type:
			return None
		
		defaults = cls.get_demo_defaults(experiment_type) or {}
		
		return {
			"experiment_type": experiment_type,
			"operation_mode": normalized,
			"policy_path": defaults.get("policy_path", ""),
			"task_description": defaults.get("task_description", "Demo"),
			"repo_id": defaults.get("repo_id", ""),
			"root": defaults.get("root", ""),
			"fps": defaults.get("fps", 30),
			"episode_time_s": defaults.get("episode_time_s", 60),
			"reset_time_s": defaults.get("reset_time_s", 10),
			"num_episodes": defaults.get("num_episodes", 50),
			"interactive": defaults.get("interactive", True),
		}
