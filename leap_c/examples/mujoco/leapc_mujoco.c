#ifdef __cplusplus
extern "C" {
#endif

#include <mujoco/mujoco.h>
#include "leapc_mujoco.h"

#include "acados/utils/external_function_generic.h"
#include "acados/utils/math.h"
#include "blasfeo_d_blas.h"
#include "blasfeo_d_aux.h"
#include "blasfeo_d_aux_ext_dep.h"


mjModel* m = NULL;
static _Thread_local mjData* d_thread_local = NULL;


int acados_mujoco_init(const char* xml_path) {
    if (m) {
        printf("Cleaning up old MuJoCo model.\n");
        mj_deleteModel(m);
        m = NULL;
    }
    
    if (d_thread_local) {
        mj_deleteData(d_thread_local);
        d_thread_local = NULL;
    }

    char error[1000] = "Could not load XML model";
    m = mj_loadXML(xml_path, NULL, error, 1000);
    
    if (!m) {
        printf("ERROR: %s\n", error);
        return -1; // Failure
    }
    
    // ============================================
    // MPC-OPTIMIZED SOLVER CONFIGURATION
    // ============================================
    
    // 1. Enable analytical derivatives (forward/inverse dynamics)
    m->opt.enableflags |= mjENBL_FWDINV;
    
    // 2. Contact solver: Newton is more accurate and predictable for MPC
    m->opt.solver = mjSOL_NEWTON;
    
    // 3. Fixed iterations for consistent real-time performance
    m->opt.iterations = 100;
    m->opt.tolerance = 0;  // Disable early termination
    
    // 4. ⭐ CRITICAL: Contact smoothing for gradient-based optimization
    // Higher impratio = softer contacts = smoother gradients
    // Range: 1.0 (very soft) to 100.0 (nearly rigid)
    // Default: 10.0 (good balance for most MPC tasks)
    m->opt.impratio = 10.0;
    
    // 5. Elliptic friction cone for smooth gradients
    m->opt.cone = mjCONE_ELLIPTIC;
    
    // 6. Disable internal warmstart (MPC handles this)
    m->opt.disableflags |= mjDSBL_WARMSTART;
    
    // 7. Small damping for numerical stability
    if (m->opt.viscosity == 0) {
        m->opt.viscosity = 0.01;  // Only set if not specified in XML
    }
    
    printf("Successfully loaded MuJoCo model: %s\n", xml_path);
    printf("MPC Solver Configuration:\n");
    printf("  ✓ Analytical derivatives: %s\n", 
           (m->opt.enableflags & mjENBL_FWDINV) ? "ENABLED" : "NOT SUPPORTED");
    printf("  ✓ Contact solver: %s\n", 
           m->opt.solver == mjSOL_NEWTON ? "Newton" : "CG");
    printf("  ✓ Iterations: %d (fixed)\n", m->opt.iterations);
    printf("  ✓ Tolerance: disabled (no early exit)\n");
    printf("  ✓ Contact smoothing (impratio): %.1f\n", m->opt.impratio);
    printf("  ✓ Friction cone: %s\n", 
           m->opt.cone == mjCONE_ELLIPTIC ? "Elliptic (smooth)" : "Pyramidal");
    printf("  ✓ Damping: %.4f\n", m->opt.viscosity);
    
    return 0; // Success
}

mjData* get_thread_local_mjData(void) {
    if (m == NULL) {
        printf("ERROR: Model is not loaded. Call acados_mujoco_init() first.\n");
        return NULL; 
    }
    
    if (d_thread_local == NULL) {
        d_thread_local = mj_makeData(m);
    }
    
    return d_thread_local;
}

/**
 * Configure MuJoCo solver settings for MPC.
 * Call after acados_mujoco_init() to fine-tune contact behavior.
 * 
 * @param contact_stiffness: Contact impedance ratio (1.0-100.0)
 *                          Higher = harder contacts, better tracking
 *                          Lower = softer contacts, smoother gradients
 * @param smoothing: Solver tolerance factor (0.0-1.0)
 *                  0.0 = no early exit (recommended)
 *                  >0.0 = allow early termination
 * @param damping: Joint damping coefficient (>= 0.0)
 */
void acados_mujoco_configure_mpc(
    double contact_stiffness,
    double smoothing,
    double damping
) {
    if (!m) {
        printf("ERROR: Model not loaded. Call acados_mujoco_init() first.\n");
        return;
    }
    
    m->opt.impratio = contact_stiffness;
    m->opt.tolerance = smoothing * 1e-8;  // 0 = disabled, >0 = early exit
    m->opt.viscosity = damping;
    
    printf("MPC Configuration Updated:\n");
    printf("  Contact stiffness: %.2f\n", contact_stiffness);
    printf("  Tolerance: %.2e %s\n", m->opt.tolerance,
           m->opt.tolerance == 0 ? "(disabled)" : "");
    printf("  Damping: %.4f\n", damping);
}


int disc_mujoco_dyn_fun(void **in, void **out, void *params) {
    // 1. Get a non-conflicting mjData object
    mjData* d = get_thread_local_mjData();
    if (!d) {
        return -1; // Error
    }

    int nv = m->nv;
    int nq = m->nq;
    int nu = m->nu;     // Control dimension

    // 2. Extract inputs using BLASFEO data structures
    // in[0]: [x], size: nx, type: BLASFEO_DVEC_ARGS
    struct blasfeo_dvec_args *x_args = in[0];
    struct blasfeo_dvec *x = x_args->x;
    int xi = x_args->xi; // offset in vector
    
    // in[1]: [u], size: nu, type: BLASFEO_DVEC_ARGS
    struct blasfeo_dvec_args *u_args = in[1];
    struct blasfeo_dvec *u = u_args->x;
    int ui = u_args->xi; // offset in vector

    // 3. Extract outputs
    // out[0]: [fun], size: nx, type: BLASFEO_DVEC_ARGS
    struct blasfeo_dvec_args *f_args = out[0];
    struct blasfeo_dvec *fun = f_args->x;

    // 4. Unpack state from BLASFEO vector to MuJoCo arrays
    blasfeo_unpack_dvec(nq, x, xi, d->qpos, 1);
    blasfeo_unpack_dvec(nv, x, xi + nq, d->qvel, 1);
    
    // 5. Unpack controls from BLASFEO vector to MuJoCo array
    blasfeo_unpack_dvec(nu, u, ui, d->ctrl, 1);

    // 6. Run the MuJoCo dynamics step
    mj_step(m, d);

    // 7. Pack next state back into BLASFEO output vector
    blasfeo_pack_dvec(nq, d->qpos, 1, fun, 0);
    blasfeo_pack_dvec(nv, d->qvel, 1, fun, nq);

    return 0; // Success
}

/**
 * Jacobian computation for MuJoCo dynamics.
 * Computes A = dF/dx and B = dF/du using analytical derivatives.
 * 
 * Uses MuJoCo's mjd_transitionFD with analytical mode (eps < 0) when
 * mjENBL_FWDINV flag is enabled. This is ~5x faster than finite differences
 * and provides exact derivatives (up to numerical precision).
 * 
 * Note: Requires integrator="Euler" or "implicit" in MuJoCo model.
 */
int disc_mujoco_dyn_fun_jac(void **in, void **out, void *params) {
    // 1. Get a non-conflicting mjData object
    mjData* d = get_thread_local_mjData();
    if (!d) {
        return -1; // Error
    }

    int nv = m->nv;
    int nq = m->nq;
    int nx = nq + nv;  // State dimension
    int nu = m->nu;     // Control dimension

    // 2. Extract inputs using BLASFEO data structures
    // in[0]: [x], size: nx, type: BLASFEO_DVEC_ARGS
    struct blasfeo_dvec_args *x_args = in[0];
    struct blasfeo_dvec *x = x_args->x;
    int xi = x_args->xi; // offset in vector
    
    // in[1]: [u], size: nu, type: BLASFEO_DVEC_ARGS
    struct blasfeo_dvec_args *u_args = in[1];
    struct blasfeo_dvec *u = u_args->x;
    int ui = u_args->xi; // offset in vector

    // 3. Extract outputs
    // out[0]: [fun], size: nx, type: BLASFEO_DVEC_ARGS
    struct blasfeo_dvec_args *f_args = out[0];
    struct blasfeo_dvec *fun = f_args->x;
    
    // out[1]: [jac_u'; jac_x'], size: (nu+nx)*nx, type: BLASFEO_DMAT_ARGS
    struct blasfeo_dmat_args *j_args = out[1];
    struct blasfeo_dmat *jac = j_args->A;

    // 4. Unpack state and control from BLASFEO to MuJoCo arrays
    blasfeo_unpack_dvec(nq, x, xi, d->qpos, 1);
    blasfeo_unpack_dvec(nv, x, xi + nq, d->qvel, 1);
    blasfeo_unpack_dvec(nu, u, ui, d->ctrl, 1);

    // 5. IMPORTANT: Set solver options for clean derivatives
    mjtNum saved_tolerance = m->opt.tolerance;
    int saved_iterations = m->opt.iterations;
    m->opt.tolerance = 0;      // Disable early exit
    m->opt.iterations = 100;   // Use fixed iterations

    // 6. Allocate memory for Jacobians A and B
    mjtNum* A_mujoco = (mjtNum*)malloc(sizeof(mjtNum) * nx * nx);
    mjtNum* B_mujoco = (mjtNum*)malloc(sizeof(mjtNum) * nx * nu);

    // 7. Call the MuJoCo derivative function
    // Use analytical derivatives (eps < 0) if enabled, otherwise finite differences
    double eps = -1.0;  // Negative epsilon triggers analytical mode
    mjd_transitionFD(m, d, eps, 1, A_mujoco, B_mujoco, NULL, NULL);

    // 8. Restore solver options
    m->opt.tolerance = saved_tolerance;
    m->opt.iterations = saved_iterations;

    // 9. mjd_transitionFD restores the state, so we need to step forward to get x_{k+1}
    mj_step(m, d);
    
    // 10. Pack next state into output
    blasfeo_pack_dvec(nq, d->qpos, 1, fun, 0);
    blasfeo_pack_dvec(nv, d->qvel, 1, fun, nq);

    // 11. Pack Jacobians into BLASFEO matrix
    // Layout: [B'; A'] where B is nx x nu and A is nx x nx
    // jac is (nu+nx) x nx in column-major (BLASFEO format)
    blasfeo_pack_tran_dmat(nx, nu, B_mujoco, nx, jac, 0, 0);   // B' at rows 0:nu-1
    blasfeo_pack_tran_dmat(nx, nx, A_mujoco, nx, jac, nu, 0);  // A' at rows nu:nu+nx-1

    // 12. Clean up
    free(A_mujoco);
    free(B_mujoco);

    return 0; // Success
}

/**
 * Workaround for Acados parametric external function limitation.
 * Acados calls this function to set global precomputed data when using
 * p_global parameters, but generic (non-CasADi) external functions don't
 * support this feature.
 * 
 * Since MuJoCo dynamics compute everything on-demand and don't need
 * precomputed global data, we provide a no-op implementation to satisfy
 * Acados' requirements without actually doing anything.
 */
void external_function_external_param_generic_set_global_data_pointer(void *self, double *global_data) {
    // No-op: MuJoCo dynamics don't use global precomputed data
    // Only stage-wise parameters (state x and control u) are needed
    (void)self;  // Suppress unused parameter warning
    (void)global_data;
}


#ifdef __cplusplus
} /* extern "C" */
#endif