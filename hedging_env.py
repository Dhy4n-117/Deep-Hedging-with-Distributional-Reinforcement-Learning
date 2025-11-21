import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.stats import norm

class HedgingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, mode='bsm', option_strike=100.0, r=0.0, risk_aversion=1.0):
        super().__init__()
        self.mode = mode
        self.strike = option_strike
        self.r = r
        self.risk_aversion = risk_aversion
        self.dt = 1/252
        self.T_max = 60/252

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -np.inf], dtype=np.float32),
            high=np.array([np.inf, 1.0, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )
        self.reset()

    def _get_bs_price(self, S, T, sigma):
        if T <= 0: return max(S - self.strike, 0.0)
        d1 = (np.log(S / self.strike) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - self.strike * np.exp(-self.r * T) * norm.cdf(d2)

    def _get_obs(self):
        return np.array([self.S, self.T - self.t, self.sigma, self.hedge_position], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.S = 100.0
        self.T = self.T_max
        self.t = 0.0
        self.hedge_position = 0.0

        if self.mode == 'bsm':
            self.sigma = 0.2
        elif self.mode == 'heston':
            self.sigma = 0.2
            self.theta = 0.2
            self.kappa = 2.0
            self.xi = 0.3
            self.rho = -0.7

        self.option_value = self._get_bs_price(self.S, self.T, self.sigma)
        self.portfolio_value = -self.option_value
        return self._get_obs(), {}

    def step(self, action):
        trade_amt = action[0]
        transaction_cost = abs(trade_amt) * self.S * 0.001
        self.hedge_position += trade_amt

        prev_S = self.S
        prev_opt_val = self.option_value

        z1 = np.random.normal()
        z2 = np.random.normal()

        if self.mode == 'bsm':
            dS = self.S * (self.r * self.dt + self.sigma * np.sqrt(self.dt) * z1)
            self.S += dS
        elif self.mode == 'heston':
            z2_corr = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2
            var = self.sigma ** 2
            d_var = self.kappa * (self.theta**2 - var) * self.dt + self.xi * np.sqrt(var) * np.sqrt(self.dt) * z2_corr
            new_var = max(0.001, var + d_var)
            self.sigma = np.sqrt(new_var)
            dS = self.S * (self.r * self.dt + self.sigma * np.sqrt(self.dt) * z1)
            self.S += dS

        self.t += self.dt
        time_remaining = self.T - self.t
        self.option_value = self._get_bs_price(self.S, time_remaining, self.sigma)

        option_pnl = -(self.option_value - prev_opt_val)
        stock_pnl = self.hedge_position * (self.S - prev_S)
        daily_pnl = option_pnl + stock_pnl - transaction_cost

        reward = daily_pnl - (self.risk_aversion * (daily_pnl**2))

        terminated = time_remaining <= 0
        truncated = False
        info = {"daily_pnl": daily_pnl, "stock_price": self.S}
        return self._get_obs(), reward, terminated, truncated, info
