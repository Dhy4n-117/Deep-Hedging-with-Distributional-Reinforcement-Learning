import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import TQC
from hedging_env import HedgingEnv  # Import the class from the file

# 1. Load Environment & Agent
print("🔄 Loading Agent...")
env = DummyVecEnv([lambda: HedgingEnv(mode='heston')])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

# Load the model (Ensure 'deep_hedging_agent.zip' is in the same folder)
model = TQC.load("deep_hedging_agent", env=env)

# 2. Run Benchmark Loop
print("⚔️ Starting Benchmark: AI vs Black-Scholes...")
ai_pnls = []

for i in range(1000):
    obs = env.reset()
    done = False
    total_ai_pnl = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        info_dict = info[0]
        total_ai_pnl += info_dict['daily_pnl']

    ai_pnls.append(total_ai_pnl)

# 3. Plot Results
plt.figure(figsize=(10, 6))
plt.hist(ai_pnls, bins=50, alpha=0.7, label='Deep Hedging AI', color='blue', edgecolor='black')
plt.axvline(np.mean(ai_pnls), color='red', linestyle='dashed', linewidth=2, label=f'Mean: ${np.mean(ai_pnls):.2f}')
plt.title('Deep Hedging P&L Distribution (1000 Episodes)')
plt.xlabel('Final P&L ($)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"✅ AI Mean P&L: ${np.mean(ai_pnls):.2f}")
print(f"✅ AI Std Dev (Risk): ${np.std(ai_pnls):.2f}")
