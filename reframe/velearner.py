import torch
from torch import nn
import tinycudann as tcnn
import numpy as np
from .utils import MLP

class velearner(nn.Module):
    def __init__(self,bound,device = 'cpu',opt=None):
        super().__init__()
        self.L = opt.L
        self.bound = bound

        F = 2;  N_min = 16
        if self.bound > 1:
            N_max = 2048*self.bound            
            log2_T = 20
        else:
            N_max = 512  
            log2_T = 19
        b = np.exp(np.log(N_max/N_min)/(self.L-1))
        print(f'velearner GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={self.L}')
        self.xyz_encoder = \
            tcnn.Encoding(
                n_input_dims=3, 
                encoding_config={
                    "otype": "Grid",
	                "type": "Hash",
                    "n_levels": self.L,
                    "n_features_per_level": F,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,
                    "per_level_scale": b,
                    "interpolation": "Linear"
                },
            )

        self.veoff_net = MLP(self.L*2, 3, 64, 2, bias=True,weight_norm=True).to(device)
        

    def forward(self, x, **kwargs):
        x = (x+self.bound)/(2*self.bound)
        h = self.xyz_encoder(x).float()
        veoffset = self.veoff_net(h)
        veoffset = torch.tanh(veoffset)
        
        return veoffset
