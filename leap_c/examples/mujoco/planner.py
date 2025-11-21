from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
        self.cfg = MujocoControllerConfig() if cfg is None else cfg
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

        diff_mpc = AcadosDiffMpcTorch(
            ocp,
            export_directory=export_directory,
        )
        super().__init__(param_manager=param_manager, diff_mpc=diff_mpc)
