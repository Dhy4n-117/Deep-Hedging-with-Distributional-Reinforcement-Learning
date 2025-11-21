import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TransformerExtractor(BaseFeaturesExtractor):
    """
    A Feature Extractor that uses a Transformer to detect 
    temporal patterns (market regimes) in the observation history.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0]
        self.d_model = 64
        self.nhead = 4
        self.num_layers = 2
        
        # 1. Input Projection
        self.linear_in = nn.Linear(input_dim, self.d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = nn.Parameter(th.zeros(1, 1, self.d_model))
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # 4. Output Projection
        self.linear_out = nn.Linear(self.d_model, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Reshape for Transformer: [Batch_Size, Seq_Len=1, Input_Dim]
        x = observations.unsqueeze(1) 
        
        # Project to d_model
        x = self.linear_in(x) + self.pos_encoder
        
        # Apply Transformer
        x = self.transformer_encoder(x)
        
        # Flatten (Global Average Pooling)
        x = x.mean(dim=1)
        
        # Output
        return th.nn.functional.relu(self.linear_out(x))
