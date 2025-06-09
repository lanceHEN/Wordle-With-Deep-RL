import torch
import torch.nn as nn

# Wrapper class that combines the ObservationEncoder and SharedEncoder modules.
# This can then be combined with either a PolicyHead or ValueHead.
class ObservationSharedWrapper(nn.Module):
    
    # given an ObservationEncoder and SharedEncoder, produces an ObservationSharedWrapper
    def __init__(self, observation_encoder, shared_encoder):
        super().__init__()
        self.observation_encoder = observation_encoder
        self.shared_encoder = shared_encoder
        
    # given batched observations, produces batched latent vector representations to be fed either into policy head or value head
    # in other words, this combines the forward pass for ObservationEncoder and SharedEncoder.
    def forward(self, batched_obs):
        # batched_obs: [B x 6 x 5 x letter_embed_dim + 3]
        encoded_grid, meta_tensor = self.observation_encoder(batched_obs)
        h = self.shared_encoder.forward(encoded_grid, meta_tensor)
        return h # [B, output_dim]