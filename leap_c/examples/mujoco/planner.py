import ctypes
import glob
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from leap_c.examples.mujoco import MUJOCO_HEADER, MUJOCO_SOURCE
from leap_c.examples.mujoco.acados_ocp import (
    MujocoAcadosCostType,
    MujocoAcadosParamInterface,
    create_mujoco_params,
    export_mujoco_ocp,
)
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager
from leap_c.ocp.acados.planner import AcadosPlanner
from leap_c.ocp.acados.torch import AcadosDiffMpcCtx, AcadosDiffMpcTorch


@dataclass(kw_only=True)
class MujocoControllerConfig:
    """Configuration for the Chain controller.

    Attributes:
        N_horizon: The number of steps in the MPC horizon.
            The MPC will have N+1 nodes (the nodes 0...N-1 and the terminal
            node N).
        T_horizon: The duration of the MPC horizon. One step during planning
            will equal T_horizon/N_horizon simulation time.
        param_interface: Determines the exposed paramete interface of the
            controller.
        cost_type: The type of cost to use, either "EXTERNAL" or "NONLINEAR_LS".
        max_iter: Maximum number of iterations for the NLP solver.
    """

    N_horizon: int = 20
    T_horizon: float = 1.0
    param_interface: MujocoAcadosParamInterface = "global"
    cost_type: MujocoAcadosCostType = "NONLINEAR_LS"
    max_iter: int = 200


class MujocoAcadosPlanner(AcadosPlanner[AcadosDiffMpcCtx]):
    """Acados-based controller for a Mujoco system.

    The state corresponds to nq+nv entries and the action corresponds to nu entries. The
    cost function takes the form of a weighted least-squares cost on the full state and action and
    the dynamics correspond to the simulated ODE also found in the corresponding mujoco environment.
    The inequality constraints are box constraints on the action.

    Attributes:
        cfg: A configuration object containing high-level settings for the MPC problem,
            such as horizon length.
    """

    cfg: MujocoControllerConfig

    def __init__(
        self,
        nq: int,
        nv: int,
        nu: int,
        u_min: np.ndarray | None,
        u_max: np.ndarray | None,
        model_path: str | Path,
        cfg: MujocoControllerConfig | None = None,
        params: list[AcadosParameter] | None = None,
        export_directory: Path | None = None,
    ):
        """Initializes the ChainController.

        Args:
            nq: Number of generalized positions (MuJoCo model.nq).
            nv: Number of generalized velocities (MuJoCo model.nv).
            nu: Number of actuators (MuJoCo model.nu).
            u_min: Lower bounds on control inputs. If None, no lower bounds.
            u_max: Upper bounds on control inputs. If None, no upper bounds.
            model_path: Path to the MuJoCo XML model file.
            cfg: cfg: A configuration object containing high-level settings for the
                MPC problem, such as horizon length and maximum force. If not provided,
                a default config is used.
            params: An optional list of parameters to define the
                ocp object. If not provided, default parameters for the Chain
                system will be created based on the cfg.
            export_directory: Directory to export the acados ocp files.
        """
        print(f"Model: {model_path}")
        print("Model dimensions:")
        print(f"nq (positions): {nq}")
        print(f"nv (velocities): {nv}")
        print(f"nu (actuators): {nu}")
        print()
        print(f"Control bounds: u_min={u_min}, u_max={u_max}")
        print()

        self.cfg = MujocoControllerConfig() if cfg is None else cfg
        print(
            f"Horizon: N={self.cfg.N_horizon}, T={self.cfg.T_horizon}s "
            f"(dt={self.cfg.T_horizon / self.cfg.N_horizon}s)"
        )
        print(f"Cost type: {self.cfg.cost_type}")
        print()

        params = (
            create_mujoco_params(
                nq=nq,
                nv=nv,
                nu=nu,
                param_interface=self.cfg.param_interface,
                N_horizon=self.cfg.N_horizon,
            )
            if params is None
            else params
        )

        param_manager = AcadosParameterManager(
            parameters=params,
            N_horizon=self.cfg.N_horizon,  # type:ignore
        )

        print(f"Learnable parameters: {list(param_manager.learnable_parameters.keys())}")
        print(f"Non-learnable parameters: {list(param_manager.non_learnable_parameters.keys())}")
        print()

        # Copy the MuJoCo C source and header to current directory to avoid file conflicts
        # Acados expects the generic source file to be in the current directory
        local_source = Path("leapc_mujoco.c")
        local_header = Path("leapc_mujoco.h")
        if not local_source.exists() or local_source.resolve() != Path(MUJOCO_SOURCE).resolve():
            print(f"  Copying {MUJOCO_SOURCE} to {local_source}")
            shutil.copy2(MUJOCO_SOURCE, local_source)
        if not local_header.exists() or local_header.resolve() != Path(MUJOCO_HEADER).resolve():
            print(f"  Copying {MUJOCO_HEADER} to {local_header}")
            shutil.copy2(MUJOCO_HEADER, local_header)

        ocp = export_mujoco_ocp(
            param_manager=param_manager,
            model_path=model_path,
            u_min=u_min,
            u_max=u_max,
            cost_type=self.cfg.cost_type,
            N_horizon=self.cfg.N_horizon,
            T_horizon=self.cfg.T_horizon,
            nlp_solver_max_iter=self.cfg.max_iter,
        )

        # Set environment variables for MuJoCo include and library paths
        # This ensures the generated Makefile/CMake can find MuJoCo headers and libraries
        import mujoco as mj_module

        mujoco_path = Path(mj_module.__file__).parent

        # Add MuJoCo include path to CPPFLAGS (used by make's implicit rules)
        current_cppflags = os.environ.get("CPPFLAGS", "")
        # Add both MuJoCo include and absolute path to current directory (for leapc_mujoco.h)
        current_dir = Path.cwd()
        mujoco_include = f"-I{mujoco_path / 'include'}"
        current_include = f"-I{current_dir}"
        os.environ["CPPFLAGS"] = f"{current_cppflags} {mujoco_include} {current_include}".strip()

        # Also set CFLAGS for CMake builds
        current_cflags = os.environ.get("CFLAGS", "")
        os.environ["CFLAGS"] = f"{current_cflags} -I{mujoco_path / 'include'}".strip()

        # Add MuJoCo library path to LDFLAGS
        # Find the actual library file (may be versioned like libmujoco.3.3.2.dylib)
        mujoco_libs = glob.glob(str(mujoco_path / "libmujoco*.dylib"))
        if not mujoco_libs:
            mujoco_libs = glob.glob(str(mujoco_path / "libmujoco*.so"))

        if mujoco_libs:
            mujoco_lib = mujoco_libs[0]
            current_ldflags = os.environ.get("LDFLAGS", "")
            os.environ["LDFLAGS"] = f"{current_ldflags} {mujoco_lib}".strip()
        else:
            print("  Warning: MuJoCo library not found!")
            current_ldflags = os.environ.get("LDFLAGS", "")
            os.environ["LDFLAGS"] = f"{current_ldflags} -L{mujoco_path} -lmujoco".strip()

        # For runtime library loading on macOS
        if sys.platform == "darwin":
            current_dyld = os.environ.get("DYLD_LIBRARY_PATH", "")
            paths = [str(mujoco_path)]
            if current_dyld:
                paths.append(current_dyld)
            os.environ["DYLD_LIBRARY_PATH"] = ":".join(paths)

        print(f"MuJoCo path: {mujoco_path}")
        print(f"CPPFLAGS: {os.environ['CPPFLAGS']}")
        print(f"LDFLAGS: {os.environ['LDFLAGS']}")
        print()

        diff_mpc = AcadosDiffMpcTorch(
            ocp,
            export_directory=export_directory,
        )

        # Load the shared library
        lib_path = diff_mpc.diff_mpc_fun.forward_batch_solver.ocp_solvers[0].shared_lib_name
        acados_lib = ctypes.CDLL(lib_path)

        # Define function signature
        # NOTE: This function is defined in leapc_mujoco.c
        acados_lib.acados_mujoco_init.argtypes = [ctypes.c_char_p]
        acados_lib.acados_mujoco_init.restype = ctypes.c_int

        # Call initialization
        result = acados_lib.acados_mujoco_init(str(model_path).encode("utf-8"))
        if result != 0:
            raise RuntimeError(f"acados_mujoco_init failed with code {result}")
        print(f"  MuJoCo initialized with model: {model_path}")
        print()

        super().__init__(param_manager=param_manager, diff_mpc=diff_mpc)
