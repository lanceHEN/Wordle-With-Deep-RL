import torch
import torch.nn as nn

# given batched 3d tensors representing the game grid (word and feedback) of shape [B, max_guesses, word_length, embed_dim]
# and a batched 1d tensor representing the turn and number of candidates remaining of shape [B, 2],
# produces latent vector representations (for each batch) with the given output dimension, output_dim
# this is a simple MLP in practice, taking in the flatten grid concatenated with the 1d additional info tensor
class SharedEncoder(nn.Module):
    
    # initializes a SharedEncoder with the given embedding dimension, and hidden dimensions
    def __init__(self, embed_dim=19, hidden_dim=512, output_dim=256):
        super().__init__()
        self.input_dim = 6 * 5 * embed_dim + 2
        
        self.encoder = nn.Sequential( # Feedforward
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=False)
        )
    
    # given batched 3d tensors representing the game grid (word and feedback) of shape [B, max_guesses, word_length, embed_dim]
    # and a batched 1d tensor representing the turn and number of candidates remaining of shape [B, 2],
    # produces latent vector representations (for each batch) with the given output dimension, output_dim
    def forward(self, grid, meta):
        B = grid.shape[0]
        flat_grid = grid.view(B, -1)  # [B, 6 * 5 * embed_dim]
        x = torch.cat([flat_grid, meta], dim=-1)  # [B, input_dim]
        return self.encoder(x)  # [B, output_dim]
