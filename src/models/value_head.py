import torch
import torch.nn as nn

# A critic/value network head that takes in batched latent vectors from SharedEncoder and produces an estimate of the values for their corresponding states.
# This is useful for PPO training.
# This is a simple MLP with one linear layer - scalar output.
class ValueHead(nn.Module):
    
    # Initializes a ValueHead with the given input dim.
    def __init__(self, input_dim=256):
        super().__init__()
        self.value = nn.Linear(input_dim, 1)
     
    # Given batched vectors from SharedEncoder, produces an estimate of the values for their corresponding states.
    def forward(self, h):
        # h: [B, input_dim]
        return self.value(h).squeeze(-1)
