import torch
import torch.nn as nn

# Variation of Shared Encoder that uses a simple FFN. Given the (batched) grid tensor and meta vectors from the Observation Encoder,
# produces latent vector representations for use by the Policy or Value heads, by flattening
# the grid, concatenating it with the meta vector, and passing through two fully connected layers.
class FFNSharedEncoder(nn.Module):
    
    # Initializes a FFNSharedEncoder with the given input dimension, hidden dimension, and output dimension.
    def __init__(self, input_dim=6*5*19+2, hidden_dim=512, shared_output_dim=256):
        super().__init__()
        
        self.encoder = nn.Sequential( # Feedforward
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, shared_output_dim),
            nn.LayerNorm(shared_output_dim),
            nn.ReLU(inplace=False)
        )
    
    # Given batched 3d tensors representing the game grids (word and feedback) of shape [B, max_guesses, word_length, embed_dim]
    # and batched 1d meta tensors representing the turn and number of candidates remaining for each batch item of shape [B, 2],
    # produces latent vector representations (for each batch item) with the given output dimension, shared_output_dim.
    def forward(self, grid, meta):
        B = grid.shape[0]
        flat_grid = grid.view(B, -1)  # [B, 6 * 5 * embed_dim]
        x = torch.cat([flat_grid, meta], dim=-1)  # [B, input_dim]
        return self.encoder(x)  # [B, shared_output_dim]