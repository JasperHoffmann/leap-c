"""Script for plotting the solution map of the HVAC planner.

This script visualizes:
1. The dependency between parameters and the optimal control (u0)
2. The dependency between parameters and the derivative du0/dp

The solution map shows how the optimal control changes as a function of 
the parameter, providing insight into the controller's behavior.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.constants import convert_temperature

from leap_c.examples.hvac.env import HvacEnvConfig, StochasticThreeStateRcEnv
from leap_c.examples.hvac.planner import HvacPlanner, HvacPlannerConfig


def plot_solution_map(
    planner: HvacPlanner,
    obs: torch.Tensor,
    param_name: str = "ref_Ti",
    param_range: tuple[float, float] | None = None,
    n_points: int = 50,
    use_numerical_derivative: bool = False,
) -> dict:
    """Plot the solution map: parameter vs control and parameter vs derivative.

    Args:
        planner: The HVAC planner instance.
        obs: Observation tensor from the environment (shape: (1, obs_dim)).
        param_name: Name of the parameter to vary.
        param_range: Tuple of (min, max) values for the parameter.
            If None, uses the parameter's space bounds.
        n_points: Number of points to sample in the parameter range.
        use_numerical_derivative: If True, compute derivative numerically for comparison.

    Returns:
        Dictionary containing all plot data including trajectories.
    """
    # Get default parameters
    default_param = planner.default_param(obs.numpy())
    default_param_torch = torch.from_numpy(default_param).to(obs.device)

    # Find the index of the parameter to vary using parameter manager keys
    param_idx = None
    param_offset = 0
    learnable_param_keys = list(planner.param_manager.learnable_parameters.keys())

    for key in learnable_param_keys:
        # Get the size of this parameter entry
        param_entry = planner.param_manager.learnable_parameters[key]
        param_size = int(np.prod(param_entry.shape)) if param_entry.shape else 1

        if key == param_name or key.startswith(f"{param_name}_"):
            param_idx = param_offset
            break
        param_offset += param_size

    if param_idx is None:
        raise ValueError(
            f"Parameter '{param_name}' not found in learnable parameters. "
            f"Available: {learnable_param_keys}"
        )

    # Determine parameter range
    if param_range is None:
        param_space = planner.param_manager.parameters[param_name].space
        param_range = (float(param_space.low[0]), float(param_space.high[0]))

    # Sample parameter values
    param_values = np.linspace(param_range[0], param_range[1], n_points)

    # Arrays to store results
    u0_values = np.zeros(n_points)
    du0_dp_values = np.zeros(n_points)
    
    # Store last context for trajectory plotting
    last_ctx = None
    last_x = None

    print(f"Computing solution map for parameter '{param_name}'...")
    print(f"Parameter range: [{param_range[0]:.2f}, {param_range[1]:.2f}]")
    print(f"Number of points: {n_points}")

    # Solve for each parameter value
    for i, p_val in enumerate(param_values):
        # Create parameter tensor with the varied parameter
        param = default_param_torch.clone()
        param[0, param_idx] = p_val

        # Enable gradient computation for the parameter
        param = param.requires_grad_(True)

        # Forward pass
        ctx, u0, x, u, value = planner.forward(obs, action=None, param=param, ctx=None)

        # Store control value (u0 is qh - heater power at next step)
        u0_values[i] = u0.detach().cpu().numpy().flatten()[0]

        # Compute derivative du0/dp using the sensitivity from the solver
        du0_dp = planner.sensitivity(ctx, "du0_dp")
        if du0_dp is not None and du0_dp.size > 0:
            du0_dp_values[i] = du0_dp.flatten()[param_idx]
        
        # Store last context and trajectories (at default parameter value)
        if i == n_points // 2:  # Middle of parameter range
            last_ctx = ctx
            last_x = x.detach().cpu().numpy()

    # Compute numerical derivative for comparison if requested
    if use_numerical_derivative:
        numerical_deriv = np.gradient(u0_values, param_values)
        print(f"Max difference between analytical and numerical derivative: "
              f"{np.max(np.abs(du0_dp_values - numerical_deriv)):.6f}")

    # Warn if all analytical gradients are zero
    if np.allclose(du0_dp_values, 0):
        print("WARNING: All analytical gradients are zero! This may indicate an issue with sensitivity computation.")

    # Convert parameter values to Celsius if it's a temperature parameter
    if "Ti" in param_name or "temperature" in param_name.lower():
        param_values_plot = convert_temperature(param_values, "kelvin", "celsius")
        param_label = f"{param_name} [°C]"
    else:
        param_values_plot = param_values
        param_label = param_name

    # Return data dictionary instead of creating plots
    return {
        "param_values": param_values,
        "param_values_plot": param_values_plot,
        "param_label": param_label,
        "u0_values": u0_values,
        "du0_dp_values": du0_dp_values,
        "numerical_deriv": numerical_deriv if use_numerical_derivative else None,
        "ctx": last_ctx,
        "x": last_x,
    }


def compute_plan_for_ref(
    planner: HvacPlanner,
    obs: torch.Tensor,
    ref_Ti_celsius: float,
) -> tuple:
    """Compute a plan for a specific reference temperature.
    
    Args:
        planner: The HVAC planner instance.
        obs: Observation tensor from the environment.
        ref_Ti_celsius: Reference temperature in Celsius.
        
    Returns:
        Tuple of (ctx, x) where ctx is the context and x is the trajectory.
    """
    # Get default parameters
    default_param = planner.default_param(obs.numpy())
    default_param_torch = torch.from_numpy(default_param).to(obs.device)
    
    # Find the index of ref_Ti parameter
    param_idx = 0  # ref_Ti is the first (and only) learnable parameter
    
    # Convert reference to Kelvin and set it
    ref_Ti_kelvin = convert_temperature(ref_Ti_celsius, "celsius", "kelvin")
    param = default_param_torch.clone()
    param[0, param_idx] = ref_Ti_kelvin
    
    # Forward pass
    ctx, u0, x, u, value = planner.forward(obs, action=None, param=param, ctx=None)
    
    return ctx, x.detach().cpu().numpy()


def main():
    """Main function to demonstrate solution map plotting."""
    # Create environment to get initial observation
    env_cfg = HvacEnvConfig(
        enable_noise=False,
        randomize_params=False,
    )
    env = StochasticThreeStateRcEnv(cfg=env_cfg)

    # Create planner (only once, reused for all scenarios)
    planner_cfg = HvacPlannerConfig(
        N_horizon=env.N_forecast - 1,
        param_interface="reference",
        param_granularity="global",
    )
    planner = HvacPlanner(cfg=planner_cfg)

    print(f"\nLearnable parameters: {list(planner.param_manager.learnable_parameters.keys())}")

    # Number of scenarios to plot
    n_scenarios = 5
    N_forecast = env.N_forecast
    N_horizon = planner.cfg.N_horizon
    
    # Create figure with subplots: 7 columns
    # (qh vs p, dqh/dp vs p, Ti plans, Control plan, price, temp, solar)
    n_cols = 7
    fig, axes = plt.subplots(n_scenarios, n_cols, figsize=(22, 1.8 * n_scenarios), 
                             constrained_layout=True)
    
    # Column titles (only at the top)
    column_titles = [
        "Heater Power",
        "Sensitivity",
        "Ti Plans",
        "Control Plan (qh)",
        "Price Forecast",
        "Ambient Temp",
        "Solar Forecast",
    ]
    
    # Parameter range for ref_Ti
    param_range = (
        convert_temperature(15.0, "celsius", "kelvin"),
        convert_temperature(25.0, "celsius", "kelvin"),
    )

    for scenario_idx in range(n_scenarios):
        # Reset environment with different seed to get different initial states
        seed = 42 + scenario_idx
        obs, info = env.reset(seed=seed)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).double()

        Ti_celsius = convert_temperature(obs[2], 'kelvin', 'celsius')
        print(f"\nScenario {scenario_idx + 1}: Ti = {Ti_celsius:.2f}°C (seed={seed})")

        # Extract forecasts from observation
        # obs structure: [quarter_hour, day_of_year, Ti, Th, Te, temp_forecast..., solar_forecast..., price_forecast...]
        temp_forecast = obs[5 : 5 + N_forecast]
        solar_forecast = obs[5 + N_forecast : 5 + 2 * N_forecast]
        price_forecast = obs[5 + 2 * N_forecast : 5 + 3 * N_forecast]
        
        # Convert temperature forecast to Celsius
        temp_forecast_celsius = convert_temperature(temp_forecast, "kelvin", "celsius")

        # Compute solution map
        data = plot_solution_map(
            planner=planner,
            obs=obs_tensor,
            param_name="ref_Ti",
            param_range=param_range,
            n_points=50,
            use_numerical_derivative=True,
        )

        # Column 0: Heater Power qh vs parameter
        ax1 = axes[scenario_idx, 0]
        ax1.plot(data["param_values_plot"], data["u0_values"] / 1000, "b-", linewidth=2, label="qh")
        if scenario_idx == n_scenarios - 1:  # Only bottom row gets x-axis label
            ax1.set_xlabel(data["param_label"])
        ax1.set_ylabel("qh [kW]")
        if scenario_idx == 0:  # Only top row gets column title
            ax1.set_title(column_titles[0])
        ax1.set_xlim(data["param_values_plot"][0], data["param_values_plot"][-1])
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color="k", linestyle="--", alpha=0.5)
        
        # Add rotated scenario label on the left
        if scenario_idx == 0 or True:  # Add to all rows
            ax1.annotate(
                f"Scenario {scenario_idx + 1}",
                xy=(-0.35, 0.5),
                xycoords="axes fraction",
                fontsize=10,
                fontweight="bold",
                rotation=90,
                ha="center",
                va="center",
            )

        # Column 1: Derivative dqh/dp vs parameter
        ax2 = axes[scenario_idx, 1]
        du0_dp_kw = data["du0_dp_values"] / 1000
        ax2.plot(data["param_values_plot"], du0_dp_kw, "r-", linewidth=2, label="analytical")
        if data["numerical_deriv"] is not None:
            numerical_deriv_kw = data["numerical_deriv"] / 1000
            ax2.plot(data["param_values_plot"], numerical_deriv_kw, "g--", linewidth=1.5, 
                     label="numerical", alpha=0.7)
            ax2.legend(fontsize=8)
        if scenario_idx == n_scenarios - 1:
            ax2.set_xlabel(data["param_label"])
        ax2.set_ylabel("dqh/dp [kW/°C]")
        if scenario_idx == 0:
            ax2.set_title(column_titles[1])
        ax2.set_xlim(data["param_values_plot"][0], data["param_values_plot"][-1])
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5)

        # Get render_info and trajectories from context
        ctx = data["ctx"]
        x = data["x"]  # Shape: (batch, N+1, nx) where nx=5 [Ti, Th, Te, qh, dqh]
        render_info = ctx.render_info
        
        plan_time_steps = np.arange(N_horizon + 1)
        
        # Compute plans for different references (5 total: 15, 17, default, 23, 25°C)
        ctx_15, x_15 = compute_plan_for_ref(planner, obs_tensor, ref_Ti_celsius=15.0)
        ctx_17, x_17 = compute_plan_for_ref(planner, obs_tensor, ref_Ti_celsius=17.0)
        ctx_23, x_23 = compute_plan_for_ref(planner, obs_tensor, ref_Ti_celsius=23.0)
        ctx_25, x_25 = compute_plan_for_ref(planner, obs_tensor, ref_Ti_celsius=25.0)
        
        # Define shades of blue for different references (light to dark)
        blue_lightest = "#B0E0E6"  # Powder blue for 15°C
        blue_light = "#89CFF0"    # Light blue for 17°C
        blue_medium = "#0066CC"   # Medium blue for default (~20°C)
        blue_dark = "#00008B"     # Dark blue for 23°C
        blue_darkest = "#000033"  # Very dark blue for 25°C
        
        # Column 2: Ti planned trajectories (all references in one plot)
        ax3 = axes[scenario_idx, 2]
        
        # Plot Ti for ref=15°C
        Ti_plan_15 = convert_temperature(x_15[0, :, 0], "kelvin", "celsius")
        ax3.step(plan_time_steps, Ti_plan_15, where="post", color=blue_lightest, 
                 linewidth=1.5, label="15°C")
        
        # Plot Ti for ref=17°C
        Ti_plan_17 = convert_temperature(x_17[0, :, 0], "kelvin", "celsius")
        ax3.step(plan_time_steps, Ti_plan_17, where="post", color=blue_light, 
                 linewidth=1.5, label="17°C")
        
        # Plot Ti for default reference
        Ti_plan = convert_temperature(x[0, :, 0], "kelvin", "celsius")
        ref_Ti_value = render_info["ref_Ti"].flatten()[0]
        ax3.step(plan_time_steps, Ti_plan, where="post", color=blue_medium, 
                 linewidth=1.5, label=f"{ref_Ti_value:.0f}°C")
        
        # Plot Ti for ref=23°C
        Ti_plan_23 = convert_temperature(x_23[0, :, 0], "kelvin", "celsius")
        ax3.step(plan_time_steps, Ti_plan_23, where="post", color=blue_dark, 
                 linewidth=1.5, label="23°C")
        
        # Plot Ti for ref=25°C
        Ti_plan_25 = convert_temperature(x_25[0, :, 0], "kelvin", "celsius")
        ax3.step(plan_time_steps, Ti_plan_25, where="post", color=blue_darkest, 
                 linewidth=1.5, label="25°C")
        
        # Plot constraints (use default render_info - constraints are the same)
        lb_Ti = render_info["lb_Ti"].flatten()
        ub_Ti = render_info["ub_Ti"].flatten()
        ax3.step(np.arange(len(lb_Ti)), lb_Ti, where="post", color="red", 
                 linestyle="--", linewidth=1.0, label="lb/ub")
        ax3.step(np.arange(len(ub_Ti)), ub_Ti, where="post", color="red", 
                 linestyle="--", linewidth=1.0)
        
        if scenario_idx == n_scenarios - 1:
            ax3.set_xlabel("Time step")
        ax3.set_ylabel("Ti [°C]")
        if scenario_idx == 0:
            ax3.set_title(column_titles[2])
        ax3.set_xlim(0, N_horizon)
        ax3.legend(fontsize=5, loc="best", ncol=2, handlelength=1.0, columnspacing=0.5)
        ax3.grid(True, alpha=0.3)

        # Column 3: Control Plan (qh) - showing all five reference cases
        ax_ctrl = axes[scenario_idx, 3]
        
        # Plot qh for ref=15°C
        qh_15 = x_15[0, :, 3] / 1000
        ax_ctrl.step(plan_time_steps, qh_15, where="post", color=blue_lightest, 
                     linewidth=1.5, label="15°C")
        
        # Plot qh for ref=17°C
        qh_17 = x_17[0, :, 3] / 1000
        ax_ctrl.step(plan_time_steps, qh_17, where="post", color=blue_light, 
                     linewidth=1.5, label="17°C")
        
        # Plot qh for default reference (from solution map data)
        qh_default = x[0, :, 3] / 1000  # Convert to kW
        ax_ctrl.step(plan_time_steps, qh_default, where="post", color=blue_medium, 
                     linewidth=1.5, label=f"{ref_Ti_value:.0f}°C")
        
        # Plot qh for ref=23°C
        qh_23 = x_23[0, :, 3] / 1000
        ax_ctrl.step(plan_time_steps, qh_23, where="post", color=blue_dark, 
                     linewidth=1.5, label="23°C")
        
        # Plot qh for ref=25°C
        qh_25 = x_25[0, :, 3] / 1000
        ax_ctrl.step(plan_time_steps, qh_25, where="post", color=blue_darkest, 
                     linewidth=1.5, label="25°C")
        
        if scenario_idx == n_scenarios - 1:
            ax_ctrl.set_xlabel("Time step")
        ax_ctrl.set_ylabel("qh [kW]")
        if scenario_idx == 0:
            ax_ctrl.set_title(column_titles[3])
        ax_ctrl.set_xlim(0, N_horizon)
        ax_ctrl.legend(fontsize=5, loc="best", ncol=2, handlelength=1.0, columnspacing=0.5)
        ax_ctrl.grid(True, alpha=0.3)
        ax_ctrl.axhline(y=0, color="k", linestyle="--", alpha=0.5)

        # Column 4: Price forecast
        ax4 = axes[scenario_idx, 4]
        time_steps = np.arange(N_forecast)
        ax4.step(time_steps, price_forecast, where="post", color="purple", linewidth=1.5)
        if scenario_idx == n_scenarios - 1:
            ax4.set_xlabel("Time step")
        ax4.set_ylabel("Price [EUR/kWh]")
        if scenario_idx == 0:
            ax4.set_title(column_titles[4])
        ax4.set_xlim(0, N_forecast)
        ax4.grid(True, alpha=0.3)

        # Column 5: Temperature forecast
        ax5 = axes[scenario_idx, 5]
        ax5.step(time_steps, temp_forecast_celsius, where="post", color="orange", linewidth=1.5)
        if scenario_idx == n_scenarios - 1:
            ax5.set_xlabel("Time step")
        ax5.set_ylabel("Temp [°C]")
        if scenario_idx == 0:
            ax5.set_title(column_titles[5])
        ax5.set_xlim(0, N_forecast)
        ax5.grid(True, alpha=0.3)

        # Column 6: Solar forecast
        ax6 = axes[scenario_idx, 6]
        ax6.step(time_steps, solar_forecast, where="post", color="gold", linewidth=1.5)
        if scenario_idx == n_scenarios - 1:
            ax6.set_xlabel("Time step")
        ax6.set_ylabel("Solar [W/m²]")
        if scenario_idx == 0:
            ax6.set_title(column_titles[6])
        ax6.set_xlim(0, N_forecast)
        ax6.grid(True, alpha=0.3)

    # Save and show figure (constrained_layout handles spacing automatically)
    plt.savefig("solution_map_ref_Ti_scenarios.pdf", dpi=150, bbox_inches="tight")
    print("\nSaved: solution_map_ref_Ti_scenarios.pdf")
    plt.show()

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
