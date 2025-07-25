from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import scipy
from .config import (
    BestestHydronicParameters,
    BestestParameters,
)
from gymnasium import spaces
from scipy.constants import convert_temperature
from .util import merge_price_weather_data, transcribe_continuous_state_space

from leap_c.examples.hvac.util import (
    load_price_data,
    load_weather_data,
    transcribe_continuous_state_space,
    transcribe_discrete_state_space,
    set_temperature_limits,
)

# Constants
DAYLIGHT_START_HOUR = 6
DAYLIGHT_END_HOUR = 18
MEAN_AMBIENT_TEMPERATURE = convert_temperature(0, "celsius", "kelvin")
MAGNITUDE_AMBIENT_TEMPERATURE = 5
MAGNITUDE_SOLAR_RADIATION = 200


class StochasticThreeStateRcEnv(gym.Env):
    """
    Simulator for a three-state RC thermal model with exact discretization of Gaussian noise.

    This environment uses the matrix exponential approach to exactly discretize both the
    deterministic dynamics and the stochastic noise terms.
    """

    def __init__(
        self,
        params: None | BestestParameters = None,
        step_size: float = 900.0,  # Default 15 minutes
        start_time: pd.Timestamp | None = None,
        horizon_hours: int = 36,
        max_hours: int = 30 * 24,  # 30 days
        render_mode: str | None = None,
        price_zone: str = "NO_1",
        price_data_path: Path | None = None,
        weather_data_path: Path | None = None,
        enable_noise: bool = True,
        random_seed: int | None = None,
    ) -> None:
        """
        Initialize the stochastic environment.

        Args:
            params: Dictionary of thermal parameters
            step_size: Time step for the simulation in seconds
            ambient_temperature_function: Function for ambient temperature
            solar_radiation_function: Function for solar radiation
            enable_noise: Whether to include stochastic noise
            random_seed: Random seed for reproducibility
        """
        N_forecast = 4 * horizon_hours  # Number of forecasted ambient temperatures

        self.N_forecast = N_forecast
        self.max_steps = int(max_hours * 3600 / step_size)

        print("env N_forecast: ", self.N_forecast)

        self.obs_low = np.array(
            [
                0.0,  # quarter hour within a day
                0.0,  # day within a year
                0.0,  # Indoor temperature
                0.0,  # Radiator temperature
                0.0,  # Envelope temperature
            ]
            + [0.0] * N_forecast  # Ambient temperatures
            + [0.0] * N_forecast  # Solar radiation
            + [0.0] * N_forecast,  # Prices  TODO: Allow negative prices
            dtype=np.float32,
        )

        self.obs_high = np.array(
            [
                24 * 4 - 1,  # quarter hour within a day
                365,  # day within a year
                convert_temperature(30.0, "celsius", "kelvin"),  # Indoor temperature
                convert_temperature(500.0, "celsius", "kelvin"),  # Radiator temperature
                convert_temperature(30.0, "celsius", "kelvin"),  # Envelope temperature
            ]
            + [convert_temperature(40.0, "celsius", "kelvin")]
            * N_forecast  # Ambient temperatures
            + [MAGNITUDE_SOLAR_RADIATION] * N_forecast  # Solar radiation
            + [np.inf] * N_forecast,  # Prices
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high)

        self.action_low = np.array([-5000.0], dtype=np.float32)
        self.action_high = np.array([5000.0], dtype=np.float32)
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high)

        # Store parameters
        self.params = (
            params if params is not None else BestestHydronicParameters().to_dict()
        )

        self.step_size = step_size
        self.enable_noise = enable_noise

        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)

        # Initial state variables [K]
        self.Ti = convert_temperature(20.0, "celsius", "kelvin")
        self.Th = convert_temperature(20.0, "celsius", "kelvin")
        self.Te = convert_temperature(20.0, "celsius", "kelvin")
        self.state_0 = np.array([self.Ti, self.Th, self.Te])

        # Precompute discrete-time matrices including noise covariance
        self.Ad, self.Bd, self.Ed, self.Qd = self._compute_discrete_matrices()

        if price_data_path is None:
            price_data_path = Path(__file__).parent / "spot_prices.csv"
        if weather_data_path is None:
            weather_data_path = Path(__file__).parent / "weather.csv"

        price_data = load_price_data(csv_path=price_data_path).resample("15T").ffill()

        weather_data = (
            load_weather_data(csv_path=weather_data_path)
            .resample("15T")
            .interpolate(method="linear")
        )

        data = merge_price_weather_data(
            price_data=price_data, weather_data=weather_data, merge_type="inner"
        )



        self.data = data


        # Rename NO1 to price
        self.data.rename(
            columns={price_zone: "price", "Tout_K": "Ta", "SolGlob": "solar"},
            inplace=True,
        )

        # Drop all columns except the ones we need
        self.data = self.data[["price", "Ta", "solar"]].copy()
        self.data["price"] = self.data["price"].astype(np.float32)
        self.data["Ta"] = self.data["Ta"].astype(np.float32)
        self.data["solar"] = self.data["solar"].astype(np.float32)
        self.data["time"] = self.data.index.to_numpy(dtype="datetime64[m]")
        self.data["quarter_hour"] = (self.data.index.hour * 4 + self.data.index.minute // 15) % (24 * 4)
        self.data["day"] = self.data["time"].dt.dayofyear % 366


        self.start_time = start_time

    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation including time, state, ambient temperatures,
        solar radiation, and prices.

        Returns:
            np.ndarray: Observation vector containing temporal, state, and forecast information

        | Num | Observation                                     |
        | --- | ------------------------------------------------|
        | 0   | quarter hour of day (0-95, 15-min intervals)    |
        | 1   | day of year (0-365)                         |
        | 2   | indoor air temperature Ti [K]                   |
        | 3   | radiator temperature Th [K]                     |
        | 4   | envelope temperature Te [K]                     |
        | 5   | ambient temperature forecast t+0 [K]            |
        | 6   | ambient temperature forecast t+1 [K]            |
        | ... | ambient temperature forecast t+N-1 [K]          |
        | 5+N | solar radiation forecast t+0 [W/m²]             |
        | 6+N | solar radiation forecast t+1 [W/m²]             |
        | ... | solar radiation forecast t+N-1 [W/m²]           |
        | 5+2N| electricity price forecast t+0 [EUR/kWh]        |
        | 6+2N| electricity price forecast t+1 [EUR/kWh]        |
        | ... | electricity price forecast t+N-1 [EUR/kWh]      |

        Total observation size: 5 + 3*N_forecast

        Notes:
            - Quarter hour: 0=00:00, 1=00:15, 2=00:30, 3=00:45, ..., 95=23:45
            - All forecasts are at 15-minute intervals starting from current time
            - N_forecast is the prediction horizon length (typically 96 for 24h)
        """
        quarter_hour = self.data["quarter_hour"].iloc[self.idx]
        day_of_year = self.data["day"].iloc[self.idx]

        price_forecast = (
            self.data["price"]
            .iloc[self.idx : self.idx + self.N_forecast]
            .to_numpy()
        )

        # TODO: Implement forecasts for weather that is not a perfect copy of the data
        ambient_temperature_forecast = (
            self.data["Ta"].iloc[self.idx : self.idx + self.N_forecast].to_numpy()
        )
        solar_forecast = (
            self.data["solar"]
            .iloc[self.idx : self.idx + self.N_forecast]
            .to_numpy()
        )

        return np.concatenate(
            [
                np.array([quarter_hour, day_of_year], dtype=np.float32),
                self.state.astype(np.float32),
                ambient_temperature_forecast.astype(np.float32),
                solar_forecast.astype(np.float32),
                price_forecast.astype(np.float32),
            ]
        )

    def _compute_discrete_matrices(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute discrete-time matrices using exact discretization via matrix exponential.
        This includes both deterministic dynamics and noise covariance.
        """
        # Create noise intensity matrix Σ from parameters
        # The stochastic terms are σᵢω̇ᵢ, σₕω̇ₕ, σₑω̇ₑ
        sigma_i = np.exp(self.params["sigmai"])
        sigma_h = np.exp(self.params["sigmah"])
        sigma_e = np.exp(self.params["sigmae"])

        # Compute continuous-time Ac
        Ac, _, _ = transcribe_continuous_state_space(
            Ac=np.zeros((3, 3)),
            Bc=np.zeros((3, 1)),
            Ec=np.zeros((3, 2)),
            params=self.params,
        )

        Qd = self._compute_noise_covariance(
            Ac=Ac,
            Sigma=np.diag([sigma_i, sigma_h, sigma_e]),
            dt=self.step_size,
        )

        # Compute discrete-time state-space matrices
        Ad, Bd, Ed = transcribe_discrete_state_space(
            Ad=np.zeros((3, 3)),
            Bd=np.zeros((3, 1)),
            Ed=np.zeros((3, 2)),
            dt=self.step_size,
            params=self.params,
        )

        return Ad, Bd, Ed, Qd

    def _compute_noise_covariance(
        self, Ac: np.ndarray, Sigma: np.ndarray, dt: float
    ) -> np.ndarray:
        """
        TODO: Check if this is correct. See, e.g., Farrell, J. Sec 4.7.2
        Compute the exact discrete-time noise covariance matrix using matrix exponential.
        Q_d = ∫₀^Δt e^(Aτ) Σ Σᵀ e^(Aᵀτ) dτ.

        Args:
            Ac: Continuous-time system matrix
            Sigma: Noise intensity matrix (diagonal)
            dt: Sampling time

        Returns:
            Qd: Discrete-time noise covariance matrix
        """
        n = Ac.shape[0]  # State dimension (3)

        # Create the augmented matrix for computing the noise covariance integral
        # [ A    Σ Σᵀ ]
        # [ 0      -Aᵀ ]
        SigmaSigmaT = Sigma @ Sigma.T

        # Augmented matrix (6x6)
        M = np.block([[Ac, SigmaSigmaT], [np.zeros((n, n)), -Ac.T]])

        # Matrix exponential of augmented system
        exp_M = scipy.linalg.expm(M * dt)

        # Extract the noise covariance from the upper-right block
        # Qd = e^(A*dt) * (upper-right block of exp_M)
        Ad = exp_M[:n, :n]
        Phi = exp_M[:n, n:]

        # The discrete-time covariance is Qd = Ad @ Phi
        return Ad @ Phi

    def _reward_function(self, state: np.ndarray, action: np.ndarray) -> float:
        """
        Compute the reward based on the current state and action.

        Args:
            action: Control input (heat input to radiator)
        Returns:
            float: Reward value
        """

        quarter_hour = self.data["quarter_hour"].iloc[self.idx]
        lb, ub = set_temperature_limits(quarter_hours=quarter_hour)

        # Reward for comfort zone compliance
        comfort_reward = int(lb <= state[0] <= ub)

        # Reward for energy saving
        energy_reward = 1.0 - np.clip(
            a=action[0] / self.action_high[0], a_min=0.0, a_max=1.0
        )

        reward = 0.5 * (comfort_reward + energy_reward)

        return reward

    def _is_terminated(self) -> bool:
        """
        Check if the current state is terminal.

        Returns:
            bool: True if terminal, False otherwise
        """
        reached_max_steps = self.step_cnter >= self.max_steps
        reached_end_of_data = self.idx >= len(self.data) - self.N_forecast

        # return reached_max_time or reached_end_of_data
        return reached_end_of_data or reached_max_steps


    def step(
        self, action: np.ndarray,
    ) -> tuple[np.ndarray, None, None, None, dict, None]:
        """
        Perform a simulation step with exact discrete-time dynamics including noise.

        Args:
            action: Control input (heat input to radiator)
            time: Current time in seconds since the start of the simulation

        Returns:
            next_state: Next state after applying control input and noise
        """
        # Get exogenous inputs
        exog = np.array(
            [
                self.data["Ta"].iloc[self.idx],  # Ambient temperature
                self.data["solar"].iloc[self.idx],  # Solar radiation
            ]
        )

        # Deterministic state update
        x_next = self.Ad @ self.state + self.Bd @ action + self.Ed @ exog

        # Add Gaussian noise if enabled
        if self.enable_noise:
            # Sample from multivariate normal distribution with exact covariance
            noise = np.random.default_rng().multivariate_normal(
                mean=np.zeros(3), cov=self.Qd
            )
            x_next += noise

        self.state = x_next
        self.idx += 1
        self.step_cnter += 1

        self.Ti, self.Th, self.Te = self.state[0], self.state[1], self.state[2]

        time_forecast = (
            self.data["time"]
            .iloc[self.idx : self.idx + self.N_forecast + 1]
            .to_numpy(dtype="datetime64[m]")
        )

        obs = self._get_observation()
        reward = self._reward_function(state=self.state, action=action)
        terminated = self._is_terminated()
        truncated = None  # We do not truncate based on time steps
        info = {"time_forecast": time_forecast}

        return obs, reward, terminated, truncated, info

    def reset(self, state_0: np.ndarray | None = None, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """Reset the model state to initial values."""
        super().reset(seed=seed)

        if state_0 is None:
            state_0 = self.state_0
        self.state = state_0.copy()

        if self.start_time is not None:
            self.idx = self.data.index.get_loc(self.start_time)
        else:
            min_start_idx = 0
            max_start_idx = len(self.data) - self.N_forecast - self.max_steps +1
            self.idx = np.random.randint(
                low=min_start_idx, high=max_start_idx
            )

        self.step_cnter = 0

        obs = self._get_observation()
        time_forecast = (
            self.data["time"]
            .iloc[self.idx : self.idx + self.N_forecast + 1]
            .to_numpy(dtype="datetime64[m]")
        )
        info = {"time_forecast": time_forecast}

        return obs, info

    def get_noise_statistics(self) -> dict:
        """
        Get statistics about the noise model.

        Returns:
            Dictionary with noise statistics
        """
        sigma_i = np.exp(self.params["sigmai"])
        sigma_h = np.exp(self.params["sigmah"])
        sigma_e = np.exp(self.params["sigmae"])

        return {
            "continuous_noise_intensities": {
                "sigma_i": sigma_i,
                "sigma_h": sigma_h,
                "sigma_e": sigma_e,
            },
            "discrete_noise_covariance": self.Qd,
            "discrete_noise_std": np.sqrt(np.diag(self.Qd)),
            "step_size": self.step_size,
        }


def decompose_observation(obs: np.ndarray) -> tuple:
    """
    Decompose the observation vector into its components.

    Args:
        obs: Observation vector from the environment.

    Returns:
        Tuple containing:
        - quarter_hour: Current quarter hour of the day (0-95)
        - day_of_year: Current day of the year (1-365)
        - Ti: Indoor air temperature in Kelvin
        - Th: Radiator temperature in Kelvin
        - Te: Envelope temperature in Kelvin
        - Ta_forecast: Ambient temperature forecast for the next N steps
        - solar_forecast: Solar radiation forecast for the next N steps
        - price_forecast: Electricity price forecast for the next N steps
    """
    if obs.ndim > 1:
        N_forecast = (obs.shape[1] - 5) // 3

        quarter_hour = obs[:, 0]
        day_of_year = obs[:, 1]
        Ti = obs[:, 2]
        Th = obs[:, 3]
        Te = obs[:, 4]

        Ta_forecast = obs[:, 5 : 5 + 1 * N_forecast]
        solar_forecast = obs[:, 5 + 1 * N_forecast : 5 + 2 * N_forecast]
        price_forecast = obs[:, 5 + 2 * N_forecast : 5 + 3 * N_forecast]

        for forecast in [
            Ta_forecast,
            solar_forecast,
            price_forecast,
        ]:
            assert forecast.shape[1] == N_forecast, (
                f"Expected {N_forecast} forecasts, got {forecast.shape[1]}"
            )

        # Cast to appropriate types
        quarter_hour = quarter_hour.astype(np.int32)
        day_of_year = day_of_year.astype(np.int32)
        Ti = Ti.astype(np.float32)
        Th = Th.astype(np.float32)
        Te = Te.astype(np.float32)
        Ta_forecast = Ta_forecast.astype(np.float32)
        solar_forecast = solar_forecast.astype(np.float32)
        price_forecast = price_forecast.astype(np.float32)

    else:
        N_forecast = (len(obs) - 5) // 3

        quarter_hour = obs[0]
        day_of_year = obs[1]
        Ti = obs[2]
        Th = obs[3]
        Te = obs[4]

        Ta_forecast = obs[5 : 5 + 1 * N_forecast]
        solar_forecast = obs[5 + 1 * N_forecast : 5 + 2 * N_forecast]
        price_forecast = obs[5 + 2 * N_forecast : 5 + 3 * N_forecast]

        for forecast in [
            Ta_forecast,
            solar_forecast,
            price_forecast,
        ]:
            assert len(forecast) == N_forecast, (
                f"Expected {N_forecast} forecasts, got {len(forecast)}"
            )


    return (
        quarter_hour,
        day_of_year,
        Ti,
        Th,
        Te,
        Ta_forecast,
        solar_forecast,
        price_forecast,
    )
