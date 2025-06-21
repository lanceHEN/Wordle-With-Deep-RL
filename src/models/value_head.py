import torch.nn as nn

# A critic/value network head that takes in batched latent vectors from SharedEncoder of shape [B, input_dim]
# and produces an estimate of the values for their corresponding states as a [B] tensor.
# This is useful for PPO training.
# This is a simple MLP with one linear layer - scalar output.
class ValueHead(nn.Module):
    
    # Initializes a ValueHead with the given input dim.
    def __init__(self, input_dim=256):
        super().__init__()
        self.value = nn.Linear(input_dim, 1)
     
    # Given batched vectors from SharedEncoder of shape [B, input_dim], produces an estimate of the values for their corresponding states, of shape [B].
    def forward(self, h):
        # h: [B, input_dim]
        return self.value(h).squeeze(-1) # [B]
