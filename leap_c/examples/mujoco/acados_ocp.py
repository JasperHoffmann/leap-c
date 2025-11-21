"""Acados OCP formulation for MuJoCo dynamics using generic external functions.

This module provides a leap-c style interface for creating parametric Acados OCPs
with MuJoCo physics models. It uses the external C functions (disc_mujoco_dyn_fun,
disc_mujoco_dyn_fun_jac) for high-performance dynamics evaluation with BLASFEO
data structures.

Example:
    >>> from pathlib import Path
    >>> from leap_c.examples.mujoco.acados_ocp import create_mujoco_params, export_mujoco_ocp
    >>> from leap_c.ocp.acados.parameters import AcadosParameterManager
    >>>
    >>> model_path = Path("test_model.xml")
    >>> params = create_mujoco_params(nq=1, nv=1, nu=1)
    >>> param_manager = AcadosParameterManager(params, N_horizon=50)
    >>> ocp = export_mujoco_ocp(param_manager, model_path)
"""

from pathlib import Path
from typing import Literal

import casadi as ca
import gymnasium as gym
import numpy as np
from acados_template import AcadosOcp

from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager

MujocoAcadosParamInterface = Literal["global", "stagewise"]
"""Determines the exposed parameter interface of the controller.
"global" means that learnable parameters are the same for all stages of the horizon,
while "stagewise" means that learnable parameters can vary between stages.
"""
MujocoAcadosCostType = Literal["EXTERNAL", "NONLINEAR_LS"]
"""The type of cost to use, either "EXTERNAL" or "NONLINEAR_LS". Both model the same cost function, 
but the former uses an exact Hessian in the optimization, while the latter uses a 
Gauss-Newton Hessian approximation.
"""


def create_mujoco_params(
    nq: int,
    nv: int,
    nu: int,
    param_interface: MujocoAcadosParamInterface = "global",
    N_horizon: int = 50,
) -> list[AcadosParameter]:
    """Returns a list of parameters used in the MuJoCo controller.

    Args:
        nq: Number of generalized positions (MuJoCo model.nq).
        nv: Number of generalized velocities (MuJoCo model.nv).
        nu: Number of actuators (MuJoCo model.nu).
        param_interface: Determines the exposed parameter interface of the controller.
        N_horizon: The number of steps in the MPC horizon.

    Returns:
        List of AcadosParameter objects defining the OCP parameters.
    """
    # Default reference: upright position (q=0), zero velocity, zero control
    default_qref = np.zeros(nq)
    default_vref = np.zeros(nv)
    default_uref = np.zeros(nu)

    # Default cost weights (diagonal sqrt of Q and R matrices)
    default_q_diag_sqrt = 1 * np.ones(nq)  # Position tracking
    default_r_diag_sqrt = 0.1 * np.ones(nv)  # Velocity tracking
    default_u_diag_sqrt = 0.01 * np.ones(nu)  # Control effort

    return [
        # Cost matrix factorization parameters (learnable)
        AcadosParameter(
            "q_diag_sqrt",
            default=default_q_diag_sqrt,
            space=gym.spaces.Box(
                low=0.0,
                high=100.0,
                shape=(nq,),
                dtype=np.float64,
            ),
            interface="learnable",
            end_stages=list(range(N_horizon + 1)) if param_interface == "stagewise" else [],
        ),
        AcadosParameter(
            "r_diag_sqrt",
            default=default_r_diag_sqrt,
            space=gym.spaces.Box(
                low=0.0,
                high=100.0,
                shape=(nv,),
                dtype=np.float64,
            ),
            interface="learnable",
            end_stages=list(range(N_horizon + 1)) if param_interface == "stagewise" else [],
        ),
        AcadosParameter(
            "u_diag_sqrt",
            default=default_u_diag_sqrt,
            space=gym.spaces.Box(
                low=0.0,
                high=100.0,
                shape=(nu,),
                dtype=np.float64,
            ),
            interface="learnable",
            end_stages=list(range(N_horizon + 1)) if param_interface == "stagewise" else [],
        ),
        # Reference parameters (learnable for flexibility)
        AcadosParameter(
            "qref",
            default=default_qref,
            space=gym.spaces.Box(
                low=-10.0 * np.ones(nq),
                high=10.0 * np.ones(nq),
                dtype=np.float64,
            ),
            interface="learnable",
            end_stages=list(range(N_horizon + 1)) if param_interface == "stagewise" else [],
        ),
        AcadosParameter(
            "vref",
            default=default_vref,
            space=gym.spaces.Box(
                low=-10.0 * np.ones(nv),
                high=10.0 * np.ones(nv),
                dtype=np.float64,
            ),
            interface="learnable",
            end_stages=list(range(N_horizon + 1)) if param_interface == "stagewise" else [],
        ),
        AcadosParameter(
            "uref",
            default=default_uref,
            interface="learnable",
            end_stages=list(range(N_horizon + 1)) if param_interface == "stagewise" else [],
        ),
    ]


def export_mujoco_ocp(
    param_manager: AcadosParameterManager,
    model_path: Path | str,
    cost_type: MujocoAcadosCostType = "NONLINEAR_LS",
    name: str = "mujoco",
    N_horizon: int = 20,
    T_horizon: float = 1.0,
    x0: np.ndarray | None = None,
    u_min: np.ndarray | None = None,
    u_max: np.ndarray | None = None,
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM",
    nlp_solver_max_iter: int = 200,
) -> AcadosOcp:
    """Export a parametric Acados OCP for MuJoCo dynamics.

    This function creates an Acados OCP that uses external C functions for dynamics
    evaluation. The dynamics are computed using the leapc_mujoco.c implementation,
    which provides high-performance MuJoCo integration with BLASFEO data structures.

    Args:
        param_manager: Manager for OCP parameters (costs, references).
        model_path: Path to the MuJoCo XML model file (str or Path).
        cost_type: Cost function type ("EXTERNAL" for exact Hessian,
            "NONLINEAR_LS" for Gauss-Newton).
        name: Name for the Acados model and generated code.
        N_horizon: Number of shooting intervals in the horizon.
        T_horizon: Total time horizon in seconds.
        x0: Initial state constraint. If None, defaults to zeros.
        u_min: Lower bounds on control inputs. If None, no lower bounds.
        u_max: Upper bounds on control inputs. If None, no upper bounds.
        qp_solver: QP solver to use (default: PARTIAL_CONDENSING_HPIPM).
        nlp_solver_max_iter: Maximum number of SQP iterations.

    Returns:
        Configured AcadosOcp object ready for solver generation.

    Note:
        The model dimensions (nq, nv, nu) are automatically inferred from the parameter manager.
        Make sure to call acados_mujoco_init() before using the generated solver.
    """
    # Convert model_path to Path if it's a string
    model_path = Path(model_path)
    ocp = AcadosOcp()

    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = T_horizon

    param_manager.assign_to_ocp(ocp)

    # Infer dimensions from parameter manager
    # The params are: q_diag_sqrt, r_diag_sqrt, u_diag_sqrt, qref, vref, uref
    # We can get nq, nv, nu from the default shapes
    nq = param_manager.parameters["qref"].default.shape[0]
    nv = param_manager.parameters["vref"].default.shape[0]
    nu = param_manager.parameters["uref"].default.shape[0]
    nx = nq + nv

    ######## Model ########
    ocp.model.name = name

    ocp.dims.nx = nx
    ocp.dims.nu = nu

    ocp.model.x = ca.SX.sym("x", ocp.dims.nx)
    ocp.model.u = ca.SX.sym("u", ocp.dims.nu)

    # Generic external dynamics from leapc_mujoco.c
    # Use just the filename - Acados will look for it in the current directory
    ocp.model.dyn_ext_fun_type = "generic"
    ocp.model.dyn_generic_source = "leapc_mujoco.c"
    ocp.model.dyn_disc_fun = "disc_mujoco_dyn_fun"
    ocp.model.dyn_disc_fun_jac = "disc_mujoco_dyn_fun_jac"
    # Note: No Hessian function available
    # We use Gauss-Newton approximation for the dynamics Hessian

    ######## Cost ########
    # Build reference and cost weight vectors from parameters
    qref = param_manager.get("qref")
    vref = param_manager.get("vref")
    uref = param_manager.get("uref")
    xref = ca.vertcat(qref, vref)
    yref = ca.vertcat(xref, uref)
    yref_e = xref

    y = ca.vertcat(ocp.model.x, ocp.model.u)
    y_e = ocp.model.x

    q_diag_sqrt = param_manager.get("q_diag_sqrt")
    r_diag_sqrt = param_manager.get("r_diag_sqrt")
    u_diag_sqrt = param_manager.get("u_diag_sqrt")
    W_sqrt = ca.diag(ca.vertcat(q_diag_sqrt, r_diag_sqrt, u_diag_sqrt))
    W = W_sqrt @ W_sqrt.T
    W_e_sqrt = ca.diag(ca.vertcat(q_diag_sqrt, r_diag_sqrt))
    W_e = W_e_sqrt @ W_e_sqrt.T

    if cost_type == "EXTERNAL":
        ocp.cost.cost_type = cost_type
        ocp.model.cost_expr_ext_cost = 0.5 * (y - yref).T @ W @ (y - yref)

        ocp.cost.cost_type_e = cost_type
        ocp.model.cost_expr_ext_cost_e = 0.5 * (y_e - yref_e).T @ W_e @ (y_e - yref_e)

        # Use Gauss-Newton since we don't have dynamics Hessian
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    elif cost_type == "NONLINEAR_LS":
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        ocp.cost.W = W
        ocp.cost.yref = yref
        ocp.model.cost_y_expr = y

        ocp.cost.W_e = W_e
        ocp.cost.yref_e = yref_e
        ocp.model.cost_y_expr_e = y_e

        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    else:
        raise ValueError(f"Cost type {cost_type} not supported. Use 'EXTERNAL' or 'NONLINEAR_LS'.")

    ######## Constraints ########
    # Initial state constraint
    if x0 is None:
        x0 = np.zeros(nx)
    ocp.constraints.idxbx_0 = np.arange(nx)
    ocp.constraints.x0 = x0

    # Control bounds (if provided)
    if u_min is not None or u_max is not None:
        if u_min is None:
            u_min = -1e9 * np.ones(nu)  # Large negative value as "unbounded"
        if u_max is None:
            u_max = 1e9 * np.ones(nu)  # Large positive value as "unbounded"

        ocp.constraints.lbu = u_min
        ocp.constraints.ubu = u_max
        ocp.constraints.idxbu = np.arange(nu)

    ######## Solver configuration ########
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.qp_solver = qp_solver
    ocp.solver_options.nlp_solver_max_iter = nlp_solver_max_iter
    ocp.solver_options.tol = 1e-6
    ocp.solver_options.qp_tol = 1e-8
    ocp.solver_options.print_level = 0

    # Add MuJoCo include and library paths for compilation
    # Get MuJoCo installation path from Python package
    try:
        import sys

        import mujoco

        mujoco_path = Path(mujoco.__file__).parent

        # Add MuJoCo include path
        ocp.model.dyn_ext_fun_compile_flags = f"-I{mujoco_path / 'include'}"

        # Add MuJoCo library path (for macOS)
        if sys.platform == "darwin":
            lib_ext = "dylib"
        elif sys.platform == "linux":
            lib_ext = "so"
        else:
            lib_ext = "dll"

        mujoco_lib = mujoco_path / f"libmujoco.{lib_ext}"
        if not mujoco_lib.exists():
            # Try versioned library
            import glob

            libs = glob.glob(str(mujoco_path / f"libmujoco.*.{lib_ext}"))
            if libs:
                mujoco_lib = Path(libs[0])

        ocp.model.dyn_ext_fun_link_flags = f"-L{mujoco_path} -lmujoco"
    except ImportError:
        print("Warning: MuJoCo not found. Make sure it's installed for compilation.")

    return ocp
