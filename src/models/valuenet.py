import torch
import torch.nn as nn

# a critic/value network that takes in batched vectors from SharedEncoder and produces an estimate of the value for their corresponding states
# this is useful for PPO training
# this is a simple MLP with one linear layer - scalar output
class ValueHead(nn.Module):
    
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.value = nn.Linear(hidden_dim, 1)
     
    # given batched vectors from SharedEncoder, produces an estimate of the value for their corresponding states
    def forward(self, h):
        # h: [B, hidden_dim]
        return self.value(h).squeeze(-1)