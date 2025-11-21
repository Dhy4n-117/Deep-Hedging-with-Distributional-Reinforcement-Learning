import matplotlib.pyplot as plt
import numpy as np

# 1. Setup for a Single Episode
obs = env.reset()
done = False
dates = []
prices = []
ai_hedges = []
bs_hedges = []

day = 0
print("🎬 Replaying one month of trading...")

while not done:
    # AI Action
    action, _ = model.predict(obs, deterministic=True)

    # Get Environment Info
    # Note: We need to "peek" at the env internals to get the exact BS Delta for comparison
    # The 'env' wrapper in SB3 hides the original class, so we access it via env.envs[0]
    raw_env = env.envs[0]
    current_S = raw_env.S
    current_vol = raw_env.sigma
    time_left = raw_env.T - raw_env.t

    # Calculate Theoretical Black-Scholes Delta
    d1 = (np.log(current_S / 100) + (0.5 * current_vol ** 2) * time_left) / (current_vol * np.sqrt(time_left))
    bs_delta = norm.cdf(d1)

    # Step the environment
    obs, reward, done, info = env.step(action)

    # Store Data
    dates.append(day)
    prices.append(current_S)
    ai_hedges.append(raw_env.hedge_position) # The AI's actual held units
    bs_hedges.append(bs_delta)               # The theoretical BS target

    day += 1

# 2. Plotting The "Money Shot"
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Stock Price (Left Axis)
color = 'tab:gray'
ax1.set_xlabel('Trading Days')
ax1.set_ylabel('Stock Price ($)', color=color)
ax1.plot(dates, prices, color=color, linestyle='--', alpha=0.5, label='Stock Price')
ax1.tick_params(axis='y', labelcolor=color)

# Plot Hedges (Right Axis)
ax2 = ax1.twinx()
color_ai = 'tab:blue'
color_bs = 'tab:red'
ax2.set_ylabel('Hedge Ratio (Delta)', color='black')
ax2.plot(dates, ai_hedges, color=color_ai, linewidth=2, label='AI Hedge (Your Agent)')
ax2.plot(dates, bs_hedges, color=color_bs, linestyle=':', linewidth=2, label='Black-Scholes (Theoretical)')
ax2.tick_params(axis='y', labelcolor='black')

# Title & Legend
plt.title('Deep Hedging in Action: AI vs. Mathematical Model')
fig.tight_layout()
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
plt.grid(True, alpha=0.3)
plt.show()
