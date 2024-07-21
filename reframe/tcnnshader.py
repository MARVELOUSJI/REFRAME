import torch
from torch import nn
import numpy as np
import tinycudann as tcnn
from .utils import MLP,dir2polar

class tcnnshader(nn.Module):
    def __init__(self,bound, device = 'cpu',opt=None):
        super().__init__()
        self.bound = bound
        self.L = opt.L
        self.opt  = opt
        F = 2; N_min = 16
        if self.bound > 1:
            N_max = 2048*self.bound            
            log2_T = 20
        else:
            N_max = 512
            log2_T = 19

        b = np.exp(np.log(N_max/N_min)/(self.L-1))
        print(f'Shader GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={self.L}')
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
        self.pe_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "Frequency", 
                "n_frequencies": 6   	                     
            },
        )        
        
        self.hash_net= MLP(self.L*2, 6, 64, 1, weight_norm=False).cuda()
        
        self.rgb_net = MLP(7, 3, 64, 2,weight_norm=False).to(device)
        self.env_learner=MLP(36, 3, 256,4,weight_norm=False).to(device)
        if self.opt.region > 1:
            self.env_learner1=MLP(36, 3, 256,4,weight_norm=False).to(device)
        self._config = {
            'bound':bound,
            'device':device,
            'opt':opt,
        }
    def hashmapping(self, x):
        x = (x+self.bound)/(2*self.bound)
        h_xyz = self.xyz_encoder(x).float()
        h = self.hash_net(h_xyz)
        h = torch.sigmoid(h)
        return h


    def forward(self, x, normal, d, **kwargs):
        ndir = torch.sum(d *normal, dim=-1).unsqueeze(-1)
        wr = 2*ndir*normal-d
        h = self.hashmapping(x)
        diffuse_color = h[:,:3]
        wr_pe = self.pe_encoder(wr).float()
        if self.opt.region==1:
            env_feature = self.env_encoder(wr_pe)
        else:
            env_feature = torch.zeros(x.shape[0],3).cuda()
            indices = []
            x_min, x_max = torch.tensor(-self.opt.foreground), torch.tensor(self.opt.foreground)
            y_min, y_max = torch.tensor(-self.opt.foreground), torch.tensor(self.opt.foreground)
            z_min, z_max = torch.tensor(-self.opt.foreground), torch.tensor(self.opt.foreground)
            indices_mask = (x[:,0] >x_min) & (x[:,0] < x_max)& (x[:,1] > y_min)& (x[:,1] < y_max)& (x[:,2] > z_min)& (x[:,2] < z_max)
            indices.append(torch.nonzero(indices_mask, as_tuple=False).squeeze())  
            indices.append(torch.nonzero(~indices_mask, as_tuple=False).squeeze()) 
            env_feature[indices[0]] = self.env_learner(wr_pe[indices[0]])
            env_feature[indices[1]] = self.env_learner1(wr_pe[indices[1]])
        sepcular_color = self.rgb_net(torch.cat([env_feature,h[:,3:],ndir], 1))
        sepcular_color = torch.sigmoid(sepcular_color)
        rgbs = (sepcular_color + diffuse_color).clamp(0,1)           
        sum1 = (sepcular_color + diffuse_color - 1).clamp(0,1).mean()
         
        return sum1,rgbs,sepcular_color,diffuse_color
    
    def envmapforward(self, x, normal, d,envmap, **kwargs):
        ndir = torch.sum(d *normal, dim=-1).unsqueeze(-1)
        wr = 2*ndir*normal-d
        h = self.hashmapping(x)
        diffuse_color = h[:,:3]
        degree_xy,degree_z = dir2polar(wr)
        nordegree = torch.cat([degree_xy.unsqueeze(-1),degree_z.unsqueeze(-1)],1).double()

        env_feature = torch.nn.functional.grid_sample(envmap.double().permute(2, 1, 0).unsqueeze(0), nordegree.double().view(1, -1, 1, 2)  , mode='bilinear',padding_mode='border',align_corners=None)  
        sepcular_color = self.rgb_net(torch.cat([env_feature.float().squeeze().permute(1, 0),h[:,3:],ndir], 1))

        sepcular_color = torch.sigmoid(sepcular_color)
        rgbs = (sepcular_color + diffuse_color).clamp(0,1)      
        sum1 = (sepcular_color + diffuse_color - 1).clamp(0,1).mean()
         
        return sum1,rgbs,sepcular_color,diffuse_color
    
    @classmethod
    def load(cls,path,dev='cpu'):
        data = torch.load(path, map_location=dev)        
        shader = cls(data['config']['bound'],data['config']['device'],data['config']['opt'])
        shader.load_state_dict(data['state_dict'])
        return shader

    def save(self, path):
        data = {
            'config': self._config,
            'state_dict': self.state_dict()
        }
        torch.save(data, path)

