#!/usr/bin/env python3
"""
Wuji Hand Controller.

Talks to the wujihandros2 driver (which wraps the C++ wujihandcpp SDK)
over ROS2. Each instance owns a single hand and supports both raw joint
control and IK-based retargeting.

Multi-core parallelism: one process per hand -> independent retargeter,
independent GIL -> good multi-core CPU utilization.
"""
import logging
import time
from pathlib import Path
from typing import Callable, Optional, Tuple
import numpy as np

try:
    from wujihand_output._internal.hand_interface import WujiHand
except ImportError:
    from ._internal.hand_interface import WujiHand

try:
    from wuji_retargeting import Retargeter
    RETARGETER_AVAILABLE = True
except ImportError:
    RETARGETER_AVAILABLE = False


class WujiHandController:
    """Wuji-hand controller for a single hand.

    Talks to the wujihandros2 driver (1000Hz hardware loop). One instance
    = one hand; multi-core parallelism is provided by process-level
    isolation.

    Two control modes:
    1. Joint angle control: set the 20 joint angles directly.
    2. IK control: take a 21-point hand keypoint set and retarget it.
    """

    NUM_JOINTS = 20  # 5 fingers x 4 joints
    STARTUP_HANDOFF_DURATION_SEC = 2.0

    def __init__(
        self,
        side: str,
        hand_name: str,
        input_source: str = "manus",
        retarget_config: Optional[str] = None,
        retarget_config_dir: Optional[str] = None,
        enable_ik: bool = True,
        logger=None,
        node=None,
        retargeted_positions_callback: Optional[Callable[[np.ndarray], None]] = None,
    ):
        """
        Initialize a single-hand controller.

        Args:
            side: "left" or "right".
            hand_name: wujihandros2 driver namespace (e.g. "left_hand").
            input_source: input source type (used to pick the IK retarget config).
            retarget_config: explicit retarget config path (optional).
            retarget_config_dir: retarget config directory (optional). When
                set, lookup order is
                retarget_{input_source}_{side}.yaml then
                retarget_{input_source}.yaml — taking priority over the
                wujihand_output package's bundled config/. Used for
                multi-host deployments: launch passes the local test/
                config directory so retarget params follow the deploy host.
            enable_ik: enable IK control.
            logger: external logger.
            node: ROS2 node instance.
            retargeted_positions_callback: optional observer called with each
                valid 20-DoF IK target before the hardware startup handoff.
                This does not alter the command sent to the physical hand.
        """
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(f'WujiHandController_{side}')
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('[%(name)s] %(message)s'))
                self.logger.addHandler(handler)

        self.side = side
        self.hand_name = hand_name
        self.input_source = input_source
        self.node = node
        self._retargeted_positions_callback = retargeted_positions_callback
        self._retarget_config_dir = (
            Path(retarget_config_dir) if retarget_config_dir else None
        )

        # Wuji Glove startup handoff.  The start pose is captured from the
        # first valid wujihandros2 joint-state feedback; throughout the fixed
        # handoff window the endpoint remains the latest glove target.  MANUS
        # keeps its existing direct-control behaviour.
        self._handoff_enabled = input_source == "wuji_glove"
        self._handoff_start_positions: Optional[np.ndarray] = None
        self._handoff_started_at: Optional[float] = None
        self._handoff_complete = not self._handoff_enabled

        # Hardware interface
        self.hand: Optional[WujiHand] = None

        # IK retargeter
        self.retargeter: Optional['Retargeter'] = None
        self._ik_enabled = False

        if enable_ik and RETARGETER_AVAILABLE:
            self._init_retargeter(retarget_config)
        elif enable_ik and not RETARGETER_AVAILABLE:
            self.logger.warning("wuji_retargeting not installed; IK control unavailable")

        self._init_hand()
        self.logger.info(f"{side}-hand controller initialized (wujihandros2)")

    def _resolve_retarget_config(self) -> Optional[str]:
        """Resolve retarget_{input_source}_{side}.yaml from override dir or package share."""
        candidate_dirs = []
        if self._retarget_config_dir is not None:
            candidate_dirs.append(self._retarget_config_dir)

        try:
            from ament_index_python.packages import get_package_share_directory
            candidate_dirs.append(
                Path(get_package_share_directory("wujihand_output")) / "config"
            )
        except Exception as e:
            self.logger.warning(
                f"Could not locate wujihand_output share dir; "
                f"skipping package default retarget config: {e}")

        for cfg_dir in candidate_dirs:
            per_side = cfg_dir / f"retarget_{self.input_source}_{self.side}.yaml"
            if per_side.exists():
                self.logger.info(f"IK retarget config: {per_side}")
                return str(per_side)

        return None

    def _init_retargeter(self, config: Optional[str]) -> None:
        config_path = config or self._resolve_retarget_config()
        if config_path and Path(config_path).exists():
            self.retargeter = Retargeter.from_yaml(config_path, self.side)
            self._ik_enabled = True
            self.logger.info(f"IK retarget config: {Path(config_path).name}")
        else:
            self.logger.warning("No IK retarget config found")

    def _init_hand(self) -> None:
        if self.node is None:
            raise RuntimeError("ROS2 node was not provided")
        self.hand = WujiHand(
            hand_name=self.hand_name,
            side=self.side,
            node=self.node,
            logger=self.logger,
        )
        self.hand.connect()
        self.logger.info(f"ROS2 interface created -> /{self.hand_name}")

    # ==================== Joint-angle control ====================

    def set_joint_positions(self, positions: np.ndarray) -> bool:
        if self.hand is not None:
            return self.hand.set_joint_positions(positions)
        return False

    def get_joint_positions(self) -> Optional[np.ndarray]:
        if self.hand is not None:
            return self.hand.get_joint_positions()
        return None

    # ==================== IK control ====================

    def is_ik_available(self) -> bool:
        return self._ik_enabled

    def retarget(self, keypoints: np.ndarray) -> Optional[np.ndarray]:
        """Retarget hand keypoints to joint angles.

        Args:
            keypoints: hand keypoints, shape (21, 3) or (63,).

        Returns:
            (20,) joint-angle array, or None on failure.
        """
        if self.retargeter is None:
            return None
        try:
            keypoints = np.asarray(keypoints, dtype=np.float32)
            if keypoints.shape == (63,):
                keypoints = keypoints.reshape(21, 3)
            return self.retargeter.retarget(keypoints)
        except Exception as e:
            self.logger.error(f"{self.side}-hand IK retarget failed: {e}")
            return None

    def set_keypoints(self, keypoints: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """Drive the hand from keypoints (IK retarget + hardware command).

        Returns:
            (success, joint_angles).
        """
        angles = self.retarget(keypoints)
        if angles is None:
            return False, angles

        target = np.asarray(angles, dtype=np.float32)
        if target.shape != (self.NUM_JOINTS,) or not np.all(np.isfinite(target)):
            self.logger.error(
                f"Invalid retarget output for {self.side} hand: shape={target.shape}"
            )
            return False, angles

        # Publish/observe the raw retarget result before the hardware handoff.
        # A consumer such as a simulator must not depend on physical-hand
        # joint-state feedback being available. Keep observer failures isolated
        # so this optional path cannot interrupt the existing hardware command.
        if self._retargeted_positions_callback is not None:
            try:
                self._retargeted_positions_callback(target.copy())
            except Exception as e:
                self.logger.error(f"Failed to publish {self.side}-hand retarget output: {e}")

        if self.hand is None:
            return False, target

        if not self._handoff_complete:
            if self._handoff_start_positions is None:
                current = self.hand.get_joint_positions()
                if current is None:
                    # Do not command an unknown start pose.  The control loop
                    # will retry with the newest glove target after feedback
                    # arrives from wujihandros2.
                    return False, target
                current = np.asarray(current, dtype=np.float32)
                if (current.shape != (self.NUM_JOINTS,)
                        or not np.all(np.isfinite(current))):
                    return False, target
                self._handoff_start_positions = current.copy()
                self._handoff_started_at = time.monotonic()
                self.logger.info(
                    f"{self.side}-hand startup handoff started "
                    f"({self.STARTUP_HANDOFF_DURATION_SEC:.1f}s)"
                )

            elapsed = time.monotonic() - self._handoff_started_at
            t = min(elapsed / self.STARTUP_HANDOFF_DURATION_SEC, 1.0)
            # Quintic smoothstep: zero velocity and acceleration at both ends.
            blend = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
            command = (
                self._handoff_start_positions * (1.0 - blend)
                + target * blend
            )
            success = self.hand.set_joint_positions(command)
            if t >= 1.0 and success:
                self._handoff_complete = True
                self._handoff_start_positions = None
                self._handoff_started_at = None
                self.logger.info(
                    f"{self.side}-hand startup handoff complete; live tracking enabled"
                )
            return success, command

        success = self.hand.set_joint_positions(target)
        return success, target

    # ==================== Status & release ====================

    def is_connected(self) -> bool:
        return self.hand is not None and self.hand.is_connected()

    def disable_and_release(self) -> None:
        self.logger.info(f"Disabling {self.side} hand...")
        if self.hand is not None:
            self.hand.release()
            self.hand = None
        self.logger.info("Exited cleanly")
