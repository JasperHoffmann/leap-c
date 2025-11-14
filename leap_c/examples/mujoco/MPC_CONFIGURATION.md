# MuJoCo + Acados MPC Configuration Guide

## Default Configuration

The code is pre-configured with **best-practice defaults** for MPC:

```c
acados_mujoco_init("model.xml");
// ✓ Automatically configured with MPC-optimized settings
```

### Default Settings

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Analytical derivatives** | Enabled | 5-6x faster than finite differences |
| **Contact solver** | Newton | More accurate, predictable timing |
| **Iterations** | 100 (fixed) | No early exit → consistent timing |
| **Tolerance** | 0 (disabled) | Forces fixed iterations |
| **Contact smoothing** | 10.0 | Balanced: smooth gradients + accurate physics |
| **Friction cone** | Elliptic | Smooth approximation for gradient-based opt |
| **Warmstart** | Disabled | Let MPC handle initialization |
| **Damping** | 0.01 | Numerical stability |

These defaults work well for **most MPC applications** (locomotion, manipulation, mobile robots).

## When to Tune Settings

### Scenario 1: Optimization Struggles to Converge

**Symptoms**: SQP doesn't converge, large residuals, constraint violations

**Solution**: Make contacts softer
```c
// After init, reconfigure:
acados_mujoco_configure_mpc(
    5.0,   // contact_stiffness: softer contacts
    0.0,   // smoothing: keep fixed iterations
    0.05   // damping: more numerical stability
);
```

### Scenario 2: Poor Tracking Performance

**Symptoms**: Robot doesn't follow desired trajectory accurately

**Solution**: Make contacts stiffer
```c
acados_mujoco_configure_mpc(
    30.0,  // contact_stiffness: harder contacts
    0.0,   // smoothing: keep fixed iterations  
    0.005  // damping: less damping for responsiveness
);
```

### Scenario 3: Need Faster Solve Times

**Symptoms**: MPC can't run at required control frequency

**Solution**: Very soft contacts, looser tolerance
```c
acados_mujoco_configure_mpc(
    3.0,   // contact_stiffness: very soft
    0.5,   // smoothing: allow early termination
    0.1    // damping: high for stability
);
```

## Task-Specific Recommendations

### Locomotion (Walking, Running)
```c
// Fast footstep planning, smooth foot contacts
acados_mujoco_configure_mpc(5.0, 0.0, 0.02);
```
- Soft contacts for smooth touch-down/lift-off
- Fast convergence for high-frequency control (100-500 Hz)

### Manipulation (Grasping, In-Hand)
```c
// Precise contact forces, accurate tracking
acados_mujoco_configure_mpc(40.0, 0.0, 0.005);
```
- Stiff contacts for accurate force transmission
- Lower frequency control acceptable (50-100 Hz)

### Aerial Robots with Ground Contact
```c
// Minimal ground interaction, fast planning
acados_mujoco_configure_mpc(2.0, 0.2, 0.05);
```
- Very soft contacts (mostly flying)
- Fast solve times for agile maneuvers

### Mobile Robots (Wheeled, Tracked)
```c
// Moderate stiffness, rolling contacts
acados_mujoco_configure_mpc(10.0, 0.0, 0.01);
```
- Default settings work well
- Continuous ground contact

## Parameter Details

### Contact Stiffness (`impratio`)

**Physical meaning**: Contact impedance ratio

- `1-5`: Very compliant contacts (soft rubber, mud)
- `5-15`: **Recommended for MPC** (normal surfaces)
- `15-50`: Stiff contacts (metal, hard plastic)
- `>50`: Nearly rigid (may cause gradient noise)

**Effect on optimization**:
- Lower → Smoother Jacobians, better convergence
- Higher → More accurate physics, better tracking

### Smoothing (`tolerance`)

**Physical meaning**: When to stop constraint solver early

- `0.0`: **Recommended** - fixed 100 iterations every time
- `>0.0`: Stop early if residual < tolerance
  
**Effect**:
- `0.0` → Predictable timing (critical for real-time MPC)
- `>0.0` → Faster average time, unpredictable worst-case

### Damping (`viscosity`)

**Physical meaning**: Velocity-dependent drag force

- `0.001-0.01`: Minimal damping (fast, accurate)
- `0.01-0.05`: **Moderate** (stable, forgiving)
- `>0.05`: High damping (slow, very stable)

**Effect**:
- Lower → Less energy dissipation, potential instability
- Higher → More stable numerics, slower dynamics

## Model XML Requirements

Your MuJoCo model must be compatible:

```xml
<option integrator="Euler" timestep="0.01">
  <!--  ^^^^^^^^^ REQUIRED: Euler or implicit (not RK4) -->
</option>

<default>
  <geom friction="1.0 0.005 0.0001" 
        solimp="0.9 0.95 0.001 0.5 2"
        solref="0.02 1"/>
  <!--    ^^^^^ Lower first value = softer contacts -->
</default>
```

### Key XML Settings

- **Integrator**: Must be `Euler` or `implicit` (analytical derivatives require this)
- **Contact impedance** (`solimp`): First value < 1.0 for smooth contacts
- **Contact stiffness** (`solref`): Larger = softer contacts
- **Timestep**: 0.005-0.02 seconds typical for MPC

## Debugging Checklist

Before tuning, verify:

1. ✅ Model loads successfully
2. ✅ "Analytical derivatives: ENABLED" is printed
3. ✅ Test dynamics function (no NaN/Inf)
4. ✅ Test Jacobian accuracy (<1e-3 error for B matrix)
5. ✅ XML uses Euler integrator
6. ✅ Timestep appropriate for control frequency

If analytical derivatives aren't supported:
- Check integrator in XML (must be Euler/implicit)
- Verify MuJoCo version >= 2.3.0
- Look for error messages at model load

## Performance Guidelines

Expected solve times for 20-step horizon, 10 SQP iterations:

| Model Complexity | Jacobian (each) | Total MPC Solve | Max Frequency |
|-----------------|-----------------|-----------------|---------------|
| Simple (1-5 DOF) | 5-10 μs | 5-10 ms | 100-200 Hz |
| Medium (6-15 DOF) | 20-50 μs | 15-30 ms | 30-65 Hz |
| Complex (>15 DOF, contacts) | 100-500 μs | 50-200 ms | 5-20 Hz |

**Rule of thumb**: Jacobian computation should be <30% of total solve time.

## Example Usage

```c
// Basic usage (use defaults)
acados_mujoco_init("robot.xml");
// Ready for MPC!

// Advanced usage (custom tuning)
acados_mujoco_init("robot.xml");

// Test default settings
// ... run MPC, check convergence ...

// If needed, tune:
if (poor_convergence) {
    acados_mujoco_configure_mpc(5.0, 0.0, 0.05);  // Softer
} else if (poor_tracking) {
    acados_mujoco_configure_mpc(30.0, 0.0, 0.005); // Stiffer
}
```

## Summary

**For 90% of MPC applications**: Just use the defaults! They're carefully chosen for:
- ✅ Smooth gradients (good convergence)
- ✅ Accurate physics (good tracking)  
- ✅ Fast computation (real-time capable)
- ✅ Predictable timing (no surprises)

Only tune if you have specific requirements or observe problems.

## Quick Reference Card

```c
// Load with MPC-optimized defaults
acados_mujoco_init("model.xml");

// Optional tuning:
// acados_mujoco_configure_mpc(stiffness, smoothing, damping);

// Presets:
// Soft contacts:   (5.0,  0.0, 0.05)  - fast convergence
// Default:         (10.0, 0.0, 0.01)  - balanced
// Stiff contacts:  (30.0, 0.0, 0.005) - accurate tracking
```
