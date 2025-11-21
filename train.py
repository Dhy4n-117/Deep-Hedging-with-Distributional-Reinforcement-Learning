import numpy as np
import torch as th
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from hedging_env import HedgingEnv
from model import TransformerExtractor

# Create Environment
env = DummyVecEnv([lambda: Monitor(HedgingEnv(mode='heston'))])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

# Define Policy
policy_kwargs = dict(
    features_extractor_class=TransformerExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=[256, 256],
    n_quantiles=25
)

# Initialize Agent
model = TQC("MlpPolicy", env, top_quantiles_to_drop_per_net=2, verbose=1,
            policy_kwargs=policy_kwargs, learning_rate=3e-4, batch_size=256,
            tensorboard_log="./tensorboard_logs/")

print("🚀 Starting Training on GPU...")
model.learn(total_timesteps=100_000, log_interval=10)
model.save("deep_hedging_agent")
print("✅ DONE!")
