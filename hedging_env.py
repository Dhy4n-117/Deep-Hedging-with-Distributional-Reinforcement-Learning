import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.stats import norm

class HedgingEnv(gym.Env):
    """
    A custom Trading Environment for Deep Hedging with Transaction Costs.
    
    Dynamics:
    - 'bsm': Geometric Brownian Motion (Constant Volatility)
    - 'heston': Stochastic Volatility Model (Vol changes over time)
    
    Action Space: Continuous [-1, 1] representing the trade size.
    Observation Space: [Stock Price, Time Remaining, Volatility, Current Holdings]
    """
    
    metadata = {"render_modes": ["human"]}

    def __init__(self, mode='heston', option_strike=100.0, r=0.0, risk_aversion=1.0):
        super().__init__()
        self.mode = mode
        self.strike = option_strike
        self.r = r
        self.risk_aversion = risk_aversion
        self.dt = 1/252  # 1 trading day
        self.T_max = 60/252 # Option maturity in years (approx 2 months)
        
        # Action: Units to buy/sell (normalized to -1.0 to 1.0 range for the agent)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation: [Stock_Price, Time_to_Maturity, Current_Vol, Current_Hedge_Pos]
        # using float32 to prevent gymnasium warnings
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -np.inf], dtype=np.float32), 
            high=np.array([np.inf, 1.0, np.inf, np.inf], dtype=np.float32), 
            dtype=np.float32
        )

        self.reset()

    def _get_bs_price(self, S, T, sigma):
        """ Calculates Black-Scholes Call Price for Reward calculation """
        if T <= 0: return max(S - self.strike, 0.0)
        d1 = (np.log(S / self.strike) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - self.strike * np.exp(-self.r * T) * norm.cdf(d2)

    def _get_obs(self):
        return np.array([
            self.S, 
            self.T - self.t, 
            self.sigma, 
            self.hedge_position
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize Market State
        self.S = 100.0
        self.T = self.T_max
        self.t = 0.0
        self.hedge_position = 0.0
        
        # Volatility Initialization
        if self.mode == 'bsm':
            self.sigma = 0.2
        elif self.mode == 'heston':
            self.sigma = 0.2  # Initial vol
            self.theta = 0.2  # Long-term vol
            self.kappa = 2.0  # Mean reversion speed
            self.xi = 0.3     # Vol of Vol
            self.rho = -0.7   # Correlation between Price and Vol (Leverage Effect)

        # Track Portfolio Value for Reward
        self.option_value = self._get_bs_price(self.S, self.T, self.sigma)
        self.portfolio_value = -self.option_value # We are SHORT the option
        
        return self._get_obs(), {}

    def step(self, action):
        # 1. Parse Action
        trade_amt = action[0]
        
        # Transaction Cost (Spread + Fees): 0.1% of trade value
        transaction_cost = abs(trade_amt) * self.S * 0.001 
        
        # Update Positions
        self.hedge_position += trade_amt
        
        # 2. Market Physics (The "World" Steps Forward)
        prev_S = self.S
        prev_opt_val = self.option_value
        
        # Generate correlated random shocks
        z1 = np.random.normal()
        z2 = np.random.normal()
        
        if self.mode == 'bsm':
            # Geometric Brownian Motion
            dS = self.S * (self.r * self.dt + self.sigma * np.sqrt(self.dt) * z1)
            self.S += dS
            
        elif self.mode == 'heston':
            # Correlate the shocks
            z2_corr = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2
            
            # Stochastic Volatility (Variance process)
            var = self.sigma ** 2
            d_var = self.kappa * (self.theta**2 - var) * self.dt + self.xi * np.sqrt(var) * np.sqrt(self.dt) * z2_corr
            new_var = max(0.001, var + d_var) # Ensure variance stays positive
            self.sigma = np.sqrt(new_var)
            
            # Stock Price
            dS = self.S * (self.r * self.dt + self.sigma * np.sqrt(self.dt) * z1)
            self.S += dS

        # Advance Time
        self.t += self.dt
        time_remaining = self.T - self.t
        
        # 3. Calculate Portfolio Value & Reward
        self.option_value = self._get_bs_price(self.S, time_remaining, self.sigma)
        
        # P&L Calculation
        option_pnl = -(self.option_value - prev_opt_val) # Short position
        stock_pnl = self.hedge_position * (self.S - prev_S)
        daily_pnl = option_pnl + stock_pnl - transaction_cost
        
        # Utility Reward (Mean - Lambda * Variance)
        reward = daily_pnl - (self.risk_aversion * (daily_pnl**2))
        
        # 4. Check Termination
        terminated = time_remaining <= 0
        truncated = False
        
        # 5. Return Info (CRITICAL: This fixes the KeyError in evaluation)
        info = {
            "daily_pnl": daily_pnl, 
            "stock_price": self.S, 
            "vol": self.sigma, 
            "time": time_remaining
        }
        
        return self._get_obs(), reward, terminated, truncated, info
