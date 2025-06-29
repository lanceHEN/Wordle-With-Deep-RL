import torch
import torch.nn as nn

# Variation of Shared Encoder that uses a simple FFN. Given the (batched) grid tensor and (batched) meta tensor from the Observation Encoder,
# produces a batch of latent vector representations for use by the Policy or Value heads, by flattening
# the grid, concatenating it with the meta tensor, and passing through two fully connected layers. We use ReLU and LayerNorm.
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
    
    # Given a batched tensor representing the game grids (word and feedback) of shape [B, max_guesses, word_length, embed_dim]
    # and a batched meta tensor representing the turn and number of candidates remaining for each batch item of shape [B, 2],
    # produces a batch of latent vector representations (for each batch item) which has the overall output dimension,
    # [B, shared_output_dim], by first flattening the grid, concatenating it with the meta tensor, and passing thru the 2-layer FFN.
    def forward(self, grid, meta):
        B = grid.shape[0]
        flat_grid = grid.view(B, -1)  # [B, 6 * 5 * embed_dim]
        x = torch.cat([flat_grid, meta], dim=-1)  # [B, input_dim]
        return self.encoder(x)  # [B, shared_output_dim]
