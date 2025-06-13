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
                 conv_channels: tuple = (32, 64, 128),
                 board_hidden_dim: int = 256,    # after flattening convmap
                 first_hidden_dim: int = 512,
                 output_dim: int = 256):
        super().__init__()

        # ConvNet [C, H = 6, W = 5] over board
        in_ch = per_cell_dim
        layers = []
        for out_ch in conv_channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers) # 6×5 resolution

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
            nn.Linear(first_hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=False),
        )

    # forward pass
    def forward(self, grid, meta):
        # rearranging channel-first for Conv 2d Net [B, C, H=6, W=5]
        x = grid.permute(0, 3, 1, 2).contiguous()

        board_emb = self.readout(self.conv(x)) # [B, board_hidden_dim]
        fused = torch.cat([board_emb, meta], dim = -1)
        return self.fuse(fused) # [B, output_dim]
                     


                     



        
