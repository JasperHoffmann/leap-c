# TODO: 1. Wrap the logic already in the DiffMPCTorch interface to reduce the code required.
# TODO: 2. Clean up the code to have a c file only instead of a h file.
# TODO: 3. Set the correct model in c in the Planner class.

"""Example script demonstrating MuJoCo-Acados OCP setup and solver usage.

This script shows how to:
1. Load a MuJoCo model (from file or Gymnasium) and extract dimensions
2. Create learnable parameters for the OCP
3. Build and compile an Acados solver with MuJoCo dynamics
4. Initialize the MuJoCo C library for Acados
5. Solve an optimal control problem
6. Extract and visualize the solution

Usage:
    # Load from Gymnasium environment
    python -m leap_c.examples.mujoco.main --env Pendulum-v1
    python -m leap_c.examples.mujoco.main --env Hopper-v4 --horizon 50

    # Or load from file
    python -m leap_c.examples.mujoco.main --model leap_c/examples/mujoco/test_model.xml
    python -m leap_c.examples.mujoco.main --model /path/to/model.xml --horizon 20

Note:
    Do not run this script from the leap_c/examples/mujoco directory directly,
    as it will cause file conflicts during Acados code generation.
"""

import argparse
import os
from pathlib import Path

import numpy as np

from leap_c.examples.mujoco.env import MuJoCoEnv
from leap_c.examples.mujoco.planner import MujocoAcadosPlanner, MujocoControllerConfig


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="MuJoCo-Acados OCP Example")

    # Create mutually exclusive group for model/env specification
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        type=str,
        help="Path to MuJoCo XML model file",
    )
    model_group.add_argument(
        "--env",
        type=str,
        help="Gymnasium environment name (e.g., 'Pendulum-v1', 'Hopper-v4')",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=50,
        help="Number of shooting intervals (N)",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=2.0,
        help="Time horizon in seconds (T)",
    )
    parser.add_argument(
        "--cost_type",
        type=str,
        default="EXTERNAL",
        choices=["EXTERNAL", "NONLINEAR_LS"],
        help="Cost function type (EXTERNAL recommended for parametric costs)",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=200,
        help="Maximum SQP iterations",
    )
    parser.add_argument(
        "--init_state",
        type=str,
        default="zero",
        choices=["zero", "upright", "hanging", "qpos0"],
        help="Initial state configuration",
    )
    parser.add_argument(
        "--closed_loop",
        action="store_true",
        help="Run closed-loop MPC simulation instead of open-loop OCP",
    )
    parser.add_argument(
        "--sim_steps",
        type=int,
        default=100,
        help="Number of simulation steps for closed-loop MPC",
    )
    args = parser.parse_args()

    # Load MuJoCo environment
    if args.env:
        print("=" * 70)
        print("MuJoCo-Acados OCP Example")
        print("=" * 70)
        print(f"Loading Gymnasium environment: {args.env}")
        env = MuJoCoEnv.from_gymnasium(args.env)
        print(f"  Saved XML to: {env.xml_path}")
    else:
        model_path = Path(args.model).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print("=" * 70)
        print("MuJoCo-Acados OCP Example")
        print("=" * 70)
        print(f"Loading model from file: {model_path}")
        env = MuJoCoEnv(model_path)

    # Check if running from the mujoco directory (not recommended)
    script_dir = Path(__file__).parent.resolve()
    current_dir = Path.cwd().resolve()
    if current_dir == script_dir:
        print("⚠️  WARNING: Running from the mujoco directory may cause file conflicts.")
        print("   Recommended: Run from repository root using:")
        print(f"   python -m leap_c.examples.mujoco.main --env {args.env or args.model}")
        print()

    cfg = MujocoControllerConfig(
        N_horizon=args.horizon,
        T_horizon=args.time,
        cost_type=args.cost_type,
        max_iter=args.max_iter,
    )

    # Get control bounds from environment
    u_min, u_max = env.get_control_bounds()

    code_export_dir = os.path.abspath("./build/c_generated_code_mujoco")

    planner = MujocoAcadosPlanner(
        nq=env.nq,
        nv=env.nv,
        nu=env.nu,
        u_min=u_min,
        u_max=u_max,
        model_path=env.xml_path,
        cfg=cfg,
        export_directory=Path(code_export_dir),
    )

    ocp_solver = planner.diff_mpc.diff_mpc_fun.forward_batch_solver.ocp_solvers[0]
    # Get initial state
    x0 = env.get_initial_state(mode=args.init_state)
    print(f"Initial state ({args.init_state}): {x0}")
    print()

    # Set initial state constraint
    ocp_solver.set(0, "lbx", x0)
    ocp_solver.set(0, "ubx", x0)

    # Solve OCP
    print("Solving OCP...")
    status = ocp_solver.solve()
    sqp_iter = ocp_solver.get_stats("sqp_iter")
    time_tot = ocp_solver.get_stats("time_tot")

    print(f"  Status: {status}")
    print(f"  SQP iterations: {sqp_iter}")
    print(f"  Total time: {time_tot * 1000:.2f} ms")
    print()

    if status != 0:
        print("⚠️  WARNING: Solver did not converge to optimal solution")
        print()

    # Extract solution
    print("Solution:")
    x_traj = np.array([ocp_solver.get(i, "x") for i in range(args.horizon + 1)])
    u_traj = np.array([ocp_solver.get(i, "u") for i in range(args.horizon)])

    print(f"  Initial state: {x_traj[0]}")
    print(f"  Final state: {x_traj[-1]}")
    print(f"  First control: {u_traj[0]}")
    print(f"  Control range: [{u_traj.min():.3f}, {u_traj.max():.3f}]")
    print()

    # Print statistics
    print("Solver Statistics:")
    ocp_solver.print_statistics()
    print()

    # Run closed-loop simulation if requested
    if args.closed_loop:
        print("=" * 70)
        print("Running Closed-Loop MPC Simulation")
        print("=" * 70)
        print(f"Simulation steps: {args.sim_steps}")
        print(f"Control frequency: {1.0 / (args.time / args.horizon):.1f} Hz")
        print()

        # Create Gymnasium environment for simulation
        import time as pytime

        import gymnasium as gym

        if args.env:
            # Use specified Gymnasium environment
            gym_env = gym.make(args.env)
            print(f"Using Gymnasium environment: {args.env}")
        else:
            # Load XML file as Gymnasium environment
            # Note: This requires the XML to be compatible with Gymnasium's MuJoCo environments
            print(f"Loading model from: {args.model}")
            print("Note: Using raw XML requires manual environment wrapping")
            raise ValueError(
                "Closed-loop simulation currently requires --env flag with a "
                "Gymnasium environment name. Please use --env <env_name> instead "
                "of --model <xml_file>."
            )

        # Storage for closed-loop trajectory
        x_cl = [x0.copy()]
        u_cl = []
        solve_times = []
        statuses = []

        # Initialize simulation
        obs, info = gym_env.reset()

        gym_env.unwrapped.set_state(x0[: env.nq], x0[env.nq :])
        x_current = x0.copy()

        # Run MPC loop
        for step in range(args.sim_steps):
            # Set initial state for OCP
            ocp_solver.set(0, "lbx", x_current)
            ocp_solver.set(0, "ubx", x_current)

            # Solve OCP
            t_start = pytime.time()
            status = ocp_solver.solve()
            t_solve = pytime.time() - t_start

            # Get first control action
            u_opt = ocp_solver.get(0, "u")

            print(obs, u_opt)

            # Apply control and simulate one step forward
            obs, reward, terminated, truncated, info = gym_env.step(u_opt)

            # Extract state from environment
            qpos = gym_env.unwrapped.data.qpos.copy()
            qvel = gym_env.unwrapped.data.qvel.copy()
            x_next = np.concatenate([qpos, qvel])

            # Store results
            x_cl.append(x_next.copy())
            u_cl.append(u_opt.copy())
            solve_times.append(t_solve)
            statuses.append(status)

            # Update current state
            x_current = x_next

            # Print progress every 10 steps
            if (step + 1) % 10 == 0 or step == 0:
                print(
                    f"  Step {step + 1:3d}: status={status}, "
                    f"time={t_solve * 1000:.2f}ms, "
                    f"state=[{x_current[0]:.3f}, {x_current[1]:.3f}, ...]"
                )

        # Clean up
        gym_env.close()

        print()
        print("Closed-Loop Simulation Results:")
        print(f"  Total steps: {args.sim_steps}")
        print(f"  Successful solves: {sum(s == 0 for s in statuses)}/{args.sim_steps}")
        print(f"  Average solve time: {np.mean(solve_times) * 1000:.2f} ms")
        print(f"  Max solve time: {np.max(solve_times) * 1000:.2f} ms")
        print(f"  Final state: {x_cl[-1]}")
        print()

        # Compute tracking error if trying to stabilize at origin
        x_cl_array = np.array(x_cl)
        u_cl_array = np.array(u_cl)

        # Position tracking error (first nq states)
        pos_error = np.linalg.norm(x_cl_array[:, : env.nq], axis=1)
        print(
            f"  Position error: mean={pos_error.mean():.4f}, "
            f"max={pos_error.max():.4f}, final={pos_error[-1]:.4f}"
        )

        # Control effort
        control_effort = np.linalg.norm(u_cl_array, axis=1)
        print(f"  Control effort: mean={control_effort.mean():.4f}, max={control_effort.max():.4f}")
        print()

    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
