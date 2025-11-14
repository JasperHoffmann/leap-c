"""Lightweight MuJoCo environment wrapper for Acados OCP.

This module provides a minimal wrapper around MuJoCo models that can be used
with the Acados OCP interface. Unlike the full CartPole environment, this wrapper
focuses on providing access to model dimensions and XML paths for Gymnasium environments.

Example:
    >>> from leap_c.examples.mujoco.env import MuJoCoEnv
    >>> 
    >>> # Load from file
    >>> env = MuJoCoEnv(model_path="test_model.xml")
    >>> print(env.nq, env.nv, env.nu)
    >>> 
    >>> # Load Gymnasium environment
    >>> env = MuJoCoEnv.from_gymnasium("Pendulum-v1")
    >>> print(env.xml_path)
"""

from pathlib import Path
from typing import Literal

import gymnasium as gym
import mujoco
import numpy as np


class MuJoCoEnv:
    """Lightweight wrapper for MuJoCo models.
    
    This class provides a simple interface to MuJoCo models, extracting key dimensions
    and paths needed for Acados OCP setup. It can load models from XML files or from
    registered Gymnasium environments.
    
    Attributes:
        model: MuJoCo MjModel object.
        data: MuJoCo MjData object.
        nq: Number of generalized positions.
        nv: Number of generalized velocities.
        nu: Number of actuators.
        nx: Total state dimension (nq + nv).
        xml_path: Path to the MuJoCo XML model file.
    """
    
    def __init__(self, model_path: str | Path):
        """Initialize from a MuJoCo XML file.
        
        Args:
            model_path: Path to the MuJoCo XML model file.
        """
        self.xml_path = Path(model_path).resolve()
        if not self.xml_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.xml_path}")
        
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.model.nu
        self.nx = self.nq + self.nv
    
    @classmethod
    def from_gymnasium(cls, env_name: str) -> "MuJoCoEnv":
        """Create a MuJoCoEnv from a Gymnasium environment name.
        
        This method creates a Gymnasium environment, extracts its MuJoCo XML model,
        saves it to a temporary location, and creates a MuJoCoEnv from it.
        
        Args:
            env_name: Name of the Gymnasium environment (e.g., "Pendulum-v1", "Hopper-v4").
        
        Returns:
            MuJoCoEnv instance initialized with the Gymnasium environment's model.
        
        Raises:
            ValueError: If the environment is not a MuJoCo-based environment.
            
        Example:
            >>> env = MuJoCoEnv.from_gymnasium("Pendulum-v1")
            >>> env = MuJoCoEnv.from_gymnasium("Hopper-v4")
        """
        import tempfile
        
        # Create the Gymnasium environment
        gym_env = gym.make(env_name)
        
        # Check if it's a MuJoCo environment
        if not hasattr(gym_env.unwrapped, "model") or not isinstance(
            gym_env.unwrapped.model, mujoco.MjModel
        ):
            raise ValueError(
                f"Environment '{env_name}' is not a MuJoCo-based environment. "
                "Make sure you're using a MuJoCo environment like 'Pendulum-v1', "
                "'Hopper-v4', 'Walker2d-v4', etc."
            )
        
        # Get the model from the unwrapped environment
        mj_model = gym_env.unwrapped.model
        
        # Create a temporary directory for the XML file
        # We use NamedTemporaryFile to get a unique name, but we'll save as .xml
        temp_dir = Path(tempfile.gettempdir()) / "leap_c_mujoco_envs"
        temp_dir.mkdir(exist_ok=True)
        
        # Save the model to XML
        xml_path = temp_dir / f"{env_name.replace('/', '_')}.xml"
        mujoco.mj_saveLastXML(str(xml_path), mj_model)
        
        gym_env.close()
        
        # Fix the integrator in the XML to support forward derivatives
        # MuJoCo's RK4 integrator doesn't support finite differences for derivatives
        # We need to change it to Euler or implicit
        cls._fix_xml_integrator(xml_path)
        
        # Create and return the MuJoCoEnv
        return cls(xml_path)
    
    @staticmethod
    def _fix_xml_integrator(xml_path: Path) -> None:
        """Fix the XML integrator to support forward derivatives.
        
        MuJoCo's RK4 integrator doesn't support finite difference computation
        of derivatives (mjd_transitionFD). This method modifies the XML to use
        Euler integration instead, which is compatible with forward derivatives.
        
        Args:
            xml_path: Path to the XML file to fix.
        """
        import xml.etree.ElementTree as ET
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Find or create the option element
            option = root.find("option")
            if option is None:
                option = ET.SubElement(root, "option")
            
            # Set integrator to Euler (integrator="Euler" or integrator="implicit")
            # Euler is explicit and faster, implicit is more stable
            option.set("integrator", "Euler")
            
            # Write back the modified XML
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)
            
        except Exception as e:
            # If XML modification fails, just log a warning
            # The user can still try with the original XML
            import warnings
            warnings.warn(
                f"Failed to modify XML integrator: {e}. "
                "The model may not work with forward derivatives."
            )
    
    def get_control_bounds(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Get control input bounds from the MuJoCo model.
        
        Returns:
            Tuple of (u_min, u_max). Returns (None, None) if no bounds are specified.
        """
        if self.model.actuator_ctrlrange.any():
            u_min = self.model.actuator_ctrlrange[:, 0]
            u_max = self.model.actuator_ctrlrange[:, 1]
            return u_min, u_max
        return None, None
    
    def get_initial_state(
        self, 
        mode: Literal["zero", "upright", "hanging", "qpos0"] = "zero"
    ) -> np.ndarray:
        """Get an initial state for the system.
        
        Args:
            mode: Initial state mode:
                - "zero": All zeros
                - "upright": Position at 0, velocity at 0
                - "hanging": Position at π (for single pendulum only!)
                - "qpos0": Use model's default qpos0 with zero velocity
        
        Returns:
            Initial state vector [q; v] of shape (nx,).
            
        Note:
            For complex systems (cart-pole, etc.), "hanging" is only meaningful
            for pure pendulum systems. Use "qpos0" or "upright" for multi-DOF systems.
        """
        x0 = np.zeros(self.nx)
        
        if mode == "zero":
            pass  # Already zeros
        elif mode == "upright":
            pass  # Position 0, velocity 0
        elif mode == "hanging":
            # Only set the LAST joint to π (assumes it's the pendulum)
            # For cart-pole: cart=0, pole=π
            # For simple pendulum: pole=π
            if self.nq > 1:
                import warnings
                warnings.warn(
                    f"Using 'hanging' mode with multi-DOF system (nq={self.nq}). "
                    "Setting only the last joint to π. Consider using 'qpos0' instead."
                )
            x0[self.nq - 1] = np.pi
        elif mode == "qpos0":
            x0[:self.nq] = self.model.qpos0.copy()
        else:
            raise ValueError(f"Unknown initial state mode: {mode}")
        
        return x0
    
    def __repr__(self) -> str:
        return (
            f"MuJoCoEnv(xml_path='{self.xml_path}', "
            f"nq={self.nq}, nv={self.nv}, nu={self.nu})"
        )
