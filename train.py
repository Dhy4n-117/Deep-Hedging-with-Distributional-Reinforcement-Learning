import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import TQC
from hedging_env import HedgingEnv  # Imports your custom environment
from model import TransformerExtractor # Imports your custom brain

if __name__ == "__main__":
    # 1. Setup Environment
    env = DummyVecEnv([lambda: HedgingEnv(mode='heston')])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 2. Define Policy
    policy_kwargs = dict(
        features_extractor_class=TransformerExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[256, 256],
        n_quantiles=25
    )

    # 3. Initialize Agent
    model = TQC(
        "MlpPolicy", 
        env, 
        top_quantiles_to_drop_per_net=2, 
        verbose=1, 
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        batch_size=256,
        tensorboard_log="./tensorboard_logs/"
    )

    # 4. Train
    print("🚀 Starting Training...")
    model.learn(total_timesteps=100_000)
    
    # 5. Save
    model.save("deep_hedging_agent")
    print("✅ Saved Model.")
