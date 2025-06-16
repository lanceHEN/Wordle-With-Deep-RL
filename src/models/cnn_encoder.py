import torch
import torch.nn as nn
'''
cnnencoder.py:
    Variation of SharedEncoder that has a Convolutional Neural Network front-end.
    expects output of ObservationEncoder [B, 6, 5, per_cell_dim]
    expects meta [B, 2]
    returns [B, output_dim]
'''
class CNNSharedEncoder(nn.Module):
    '''
    given:
        grid [B, 6, 5, embed_dim]
        meta [B, 2]
    return:
    produces a fused representation using a set number of convolutions on the grid with specified channels 
    with relu activation at each step, before flattening and concatenating with the meta vector and passing
    thru an FFN
    '''
    def __init__(self,
                per_cell_dim: int = 19, # = letter_embed_dim + 3
                conv_channels: tuple = (32, 64),
                hidden_dim: int = 512,
                output_dim: int = 256):
        super().__init__()
        
        in_ch = per_cell_dim
        layers = []
        for out_ch in conv_channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        
        self.flatten = nn.Flatten()
        
        # fuses with meta tensor via Multilayer Perceptron
        self.fuse = nn.Sequential(
            nn.Linear(in_ch * 6 * 5 + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=False),
        )

    # forward pass - given grid and meta vector, produces latent vector
    def forward(self, grid, meta):
        # grid: [B, 6, 5, D]

        # rearrange so that guesses become channels (like color channels in images)
        x = grid.permute(0, 3, 2, 1).contiguous()  # [B, D, 5, 6]
        
        flat = self.flatten(self.conv(x)) # in_ch * 6 * 5

        fused = torch.cat([flat, meta], dim = -1) # in_ch * 6 * 5 + 2
        return self.fuse(fused) # [B, output_dim]
                    
                     



        
