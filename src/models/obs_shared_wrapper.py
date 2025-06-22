import torch.nn as nn
from models.observation_encoder import ObservationEncoder
from models.ffn_shared_encoder import FFNSharedEncoder
from models.letter_encoder import LetterEncoder

# Wrapper class that combines the ObservationEncoder and Shared Encoder modules into one, allowing a batch of latent vectors to be produced
# when given batched observations.
# This can then be combined with either a PolicyHead or ValueHead.
# For convenience, the ObservationEncoder and Shared Encoder do not have to be given on construction, they can be made on construction
# if set to None (we by default make an FFNSharedEncoder).
class ObservationSharedWrapper(nn.Module):
    
    def __init__(self, observation_encoder=None, shared_encoder=None):
        super().__init__()
        if observation_encoder is None:
            self.observation_encoder = ObservationEncoder()
        else:
            self.observation_encoder = observation_encoder
        
        if shared_encoder is None:
            self.shared_encoder = FFNSharedEncoder()
        else:
            self.shared_encoder = shared_encoder
        
    # Given batched observations (as a list), produces a batch of latent vector representations to be fed either into policy head or value head.
    # In other words, this combines the forward pass for ObservationEncoder and SharedEncoder.
    # The output has shape [B, shared_output_dim], where shared_output_dim is the output dimension of the SharedEncoder.
    def forward(self, batched_obs):
        encoded_grid, meta_tensor = self.observation_encoder(batched_obs)
        h = self.shared_encoder(encoded_grid, meta_tensor)
        return h # [B, shared_output_dim]
