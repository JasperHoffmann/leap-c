"""MuJoCo-Acados integration for leap-c.

This package provides high-performance MPC with MuJoCo physics models using Acados.
The C implementation uses BLASFEO data structures for efficient linear algebra
and supports analytical derivatives for gradient-based learning.

Exports:
    MUJOCO_SOURCE: Path to leapc_mujoco.c source file
    MUJOCO_HEADER: Path to leapc_mujoco.h header file
    MuJoCoEnv: Lightweight wrapper for MuJoCo models
    create_mujoco_params: Create learnable parameters for MuJoCo OCP
    export_mujoco_ocp: Export a parametric Acados OCP with MuJoCo dynamics

Example usage:
    >>> from leap_c.examples.mujoco import MuJoCoEnv, create_mujoco_params, export_mujoco_ocp
    >>> from leap_c.ocp.acados.parameters import AcadosParameterManager
    >>> 
    >>> # Load from Gymnasium
    >>> env = MuJoCoEnv.from_gymnasium("Pendulum-v1")
    >>> 
    >>> # Or from file
    >>> env = MuJoCoEnv("model.xml")
    >>> 
    >>> # Create OCP
    >>> params = create_mujoco_params(nq=env.nq, nv=env.nv, nu=env.nu)
    >>> param_manager = AcadosParameterManager(params, N_horizon=50)
    >>> ocp = export_mujoco_ocp(param_manager, env.xml_path)
"""

from pathlib import Path

from leap_c.examples.mujoco.acados_ocp import create_mujoco_params, export_mujoco_ocp
from leap_c.examples.mujoco.env import MuJoCoEnv

# Get the directory containing this file
MUJOCO_EXAMPLE_DIR = Path(__file__).parent
MUJOCO_SOURCE = MUJOCO_EXAMPLE_DIR / "leapc_mujoco.c"
MUJOCO_HEADER = MUJOCO_EXAMPLE_DIR / "leapc_mujoco.h"

__all__ = [
    "MUJOCO_SOURCE",
    "MUJOCO_HEADER",
    "MuJoCoEnv",
    "create_mujoco_params",
    "export_mujoco_ocp",
]

