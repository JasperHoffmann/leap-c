### TODOs

1. Implement Planner (usefull for reset and loading) and also move compiling there. Remove unused code:
2. Finalize Mujcoco Env
3. Make cartpole swingup work.


### Questions:

1. How does Mujoco model the dynamics exactly?
   - What integrator is used for the Mujoco envs?
2. How much should we smoothen the dynamics for improved convergence?
3. Which cost interface is flexible enough for us?
   - Reference Points?
   - Cost weighting?
4. Which algorithms are normally used for Mujoco tasks? Can we import them for comparison?
   - iLQR?
   - DDP?
   - MPPI?
