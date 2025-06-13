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
    produces a fused representation using:
        conv2d --> ReLU --> flatten
    '''
    def __init__(self,
                per_cell_dim: int = 19, # = letter_embed_dim + 3
                conv_channels: tuple = (32, 64),
                board_hidden_dim: int = 256,    # after flattening convmap
                first_hidden_dim: int = 512,
                second_hidden_dim: int = 256):
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
        
        # readout to the fixed board embedding
        self.readout = nn.Sequential(
            nn.Flatten(), # [B, in_ch * 6 * 5]
            nn.Linear(in_ch * 6 * 5, board_hidden_dim),
            nn.LayerNorm(board_hidden_dim),
            nn.ReLU(inplace=True),
        )
        # fuses with meta tensor via Multilayer Perceptron
        self.fuse = nn.Sequential(
            nn.Linear(board_hidden_dim + 2, first_hidden_dim),
            nn.LayerNorm(first_hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(first_hidden_dim, second_hidden_dim),
            nn.LayerNorm(second_hidden_dim),
            nn.ReLU(inplace=False),
        )

    # forward pass
    def forward(self, grid, meta):
        # grid: [B, 6, 5, D]

        # rearrange so that guesses become channels (like color channels in images)
        x = grid.permute(0, 3, 2, 1).contiguous()  # [B, D, 5, 6]

        board_emb = self.readout(self.conv(x)) # [B, board_hidden_dim]
        fused = torch.cat([board_emb, meta], dim = -1)
        return self.fuse(fused) # [B, output_dim]
                    
                     



        
