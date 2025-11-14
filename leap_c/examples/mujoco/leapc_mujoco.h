#ifndef LEAPC_MUJOCO_H
#define LEAPC_MUJOCO_H

#ifdef __cplusplus
extern "C" {
#endif

#include <mujoco/mujoco.h>

// Global MuJoCo model pointer (accessible to users)
extern mjModel* m;

/**
 * Initialize the MuJoCo model from an XML file.
 * Configures solver with MPC-optimized settings:
 * - Analytical derivatives enabled
 * - Newton solver with fixed iterations
 * - Contact smoothing (impratio=10.0)
 * - Elliptic friction cone
 * 
 * @param xml_path Path to the MuJoCo XML model file
 * @return 0 on success, -1 on failure
 */
int acados_mujoco_init(const char* xml_path);

/**
 * Configure MPC-specific solver settings (optional tuning).
 * Call after acados_mujoco_init() to override defaults.
 * 
 * @param contact_stiffness: Contact impedance ratio (1.0-100.0)
 *                          - Locomotion: 5-10 (soft, fast)
 *                          - Manipulation: 20-50 (stiff, accurate)
 *                          - Default: 10.0
 * 
 * @param smoothing: Solver tolerance factor (0.0-1.0)
 *                  - 0.0 = disabled (recommended for consistent timing)
 *                  - >0.0 = early termination allowed
 *                  - Default: 0.0
 * 
 * @param damping: Joint damping coefficient (>= 0.0)
 *                - Adds numerical stability
 *                - Default: 0.01
 */
void acados_mujoco_configure_mpc(
    double contact_stiffness,
    double smoothing,
    double damping
);

/**
 * Get thread-local MuJoCo data structure.
 * Each thread gets its own mjData to avoid race conditions.
 * 
 * @return Pointer to thread-local mjData, or NULL if model not initialized
 */
mjData* get_thread_local_mjData(void);

/**
 * Discrete dynamics function for Acados integration.
 * Computes x_{k+1} = F(x_k, u_k) using MuJoCo simulation.
 * 
 * Uses BLASFEO data structures:
 * - in[0]: state vector x (size: nq + nv)
 * - in[1]: control vector u (size: nu)
 * - out[0]: next state vector x_{k+1} (size: nq + nv)
 * 
 * @param in Array of input pointers (BLASFEO_DVEC_ARGS)
 * @param out Array of output pointers (BLASFEO_DVEC_ARGS)
 * @param params Optional parameters (unused)
 * @return 0 on success, -1 on error
 */
int disc_mujoco_dyn_fun(void **in, void **out, void *params);

/**
 * Discrete dynamics Jacobian function for Acados integration.
 * Computes both the function value and Jacobians A = dF/dx and B = dF/du
 * using MuJoCo's finite difference method.
 * 
 * Uses BLASFEO data structures:
 * - in[0]: state vector x (size: nq + nv)
 * - in[1]: control vector u (size: nu)
 * - out[0]: next state vector x_{k+1} (size: nq + nv)
 * - out[1]: Jacobian matrix [B'; A'] (size: (nu+nx) x nx)
 *           where B' is the transpose of B (nu x nx) and A' is transpose of A (nx x nx)
 * 
 * @param in Array of input pointers (BLASFEO_DVEC_ARGS)
 * @param out Array of output pointers (BLASFEO_DVEC_ARGS for [0], BLASFEO_DMAT_ARGS for [1])
 * @param params Optional parameters (unused)
 * @return 0 on success, -1 on error
 */
int disc_mujoco_dyn_fun_jac(void **in, void **out, void *params);

/**
 * Workaround for Acados parametric external function limitation.
 * Acados calls this function to set global precomputed data when using
 * p_global parameters, but generic (non-CasADi) external functions don't
 * natively support this feature.
 * 
 * Since MuJoCo dynamics compute everything on-demand and don't need
 * precomputed global data, we provide a no-op implementation to satisfy
 * Acados' requirements without actually doing anything.
 * 
 * @param self External function structure (unused)
 * @param global_data Pointer to global data (unused)
 */
void external_function_external_param_generic_set_global_data_pointer(void *self, double *global_data);

#ifdef __cplusplus
}
#endif

#endif // LEAPC_MUJOCO_H
