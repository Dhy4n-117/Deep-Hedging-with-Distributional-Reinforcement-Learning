import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]
        self.d_model = 64
        self.nhead = 4
        self.num_layers = 2

        self.linear_in = nn.Linear(input_dim, self.d_model)
        self.pos_encoder = nn.Parameter(th.zeros(1, 1, self.d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.linear_out = nn.Linear(self.d_model, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = observations.unsqueeze(1)
        x = self.linear_in(x) + self.pos_encoder
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return th.nn.functional.relu(self.linear_out(x))
