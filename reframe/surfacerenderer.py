import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
from .mesh import safe_normalize
from .utils import scale_img_hwc

class RandomSamples(object):
    def __init__(self, h, w, percentage=.5):
        self.idx = torch.randperm(h*w)[:int(h*w*percentage)]

    def __call__(self, tensor):
        return tensor.view(-1, tensor.shape[-1])[self.idx]

class SurfaceRenderer:
    def __init__(self, device, opt,h0,w0,mode):
        # Choose Cuda Context if your OpenGL Context doesn't work.
        self.glctx = dr.RasterizeGLContext()
        # self.glctx = dr.RasterizeCudaContext()
        self.device = device
        self.opt = opt
        self.h0 = h0
        self.w0 = w0
        self.mode = mode
        if self.opt.ssaa > 1:
            self.h = int(h0 * self.opt.ssaa)
            self.w = int(w0 * self.opt.ssaa)
        else:
            self.h, self.w = h0, w0
        print(f'rendering image resolution[{self.h},{self.w}]')
        self.full = torch.zeros(self.h * self.w, 3, dtype=torch.float32).to(self.device)
        self.dif = torch.zeros(self.h * self.w, 3, dtype=torch.float32).to(self.device)
        self.spe = torch.zeros(self.h * self.w, 3, dtype=torch.float32).to(self.device)
        self.normal_img = torch.zeros(self.h * self.w, 3, dtype=torch.float32).to(self.device)

    #Reference to NeRF2Mesh code. 
    def render(self,dataset,data, mesh, channels, shader,shading = True,envmap=None):
        if self.opt.shading_percentage < 1:
            sample_fn = RandomSamples(self.h0, self.w0, self.opt.shading_percentage)
        else:
            sample_fn = lambda x: x

        imgs_full = []
        imgs_spe = []
        imgs_diff = []
        masks = []    
        normals = []
        prefix = [self.h0,self.w0]
        for i, index in enumerate(data['index']):
            #Project mesh vertices
            pos = torch.matmul(F.pad(mesh.vertices, pad=(0, 1), mode='constant', value=1.0), torch.transpose(data['mvp'][i], 0, 1)).float().unsqueeze(0) # [1, N, 4]
            idx = mesh.indices.int()
            rast, rast_out_db = dr.rasterize(self.glctx, pos, idx, (self.h,self.w))
            
            if "mask" in channels:
                mask, _ = dr.interpolate(torch.ones_like(mesh.vertices[:, :1]).unsqueeze(0), rast, idx) 

            if "position" in channels or "depth" in channels:
                position, _ = dr.interpolate(mesh.vertices[None, ...], rast, idx)

            if "normal" in channels:
                normal, _ = dr.interpolate(mesh.vertex_normals[None, ...], rast, idx)
           
            mask_flatten = (mask > 0).view(-1).detach()
            
            position = position.view(-1, 3)
            normal = normal.view(-1, 3)
            position = sample_fn(position[mask_flatten])      
            normal = sample_fn(normal[mask_flatten])
            

            view_direction = (data['camera_location'][i] - position).cuda() #N*3
        
            view_direction = safe_normalize(view_direction)
            normal = safe_normalize(normal)
            
            if self.mode == 'train':
                full = torch.zeros(self.h * self.w, 3, dtype=torch.float32).to(self.device)
                dif = torch.zeros(self.h * self.w, 3, dtype=torch.float32).to(self.device)
                spe = torch.zeros(self.h * self.w, 3, dtype=torch.float32).to(self.device)
                normal_img = torch.zeros(self.h * self.w, 3, dtype=torch.float32).to(self.device)
            elif self.mode == 'test':
                full = self.full.zero_()
                dif = self.dif.zero_()
                spe = self.spe.zero_()
                normal_img = self.normal_img.zero_()

            if shading:
                if torch.is_tensor(envmap):
                    sum1,color,specular,diffuse = shader.envmapforward(position, normal, view_direction,envmap)
                else:
                    if self.mode == 'train':
                        all_sum1 = []
                        all_color = []
                        all_specular = []
                        all_diffuse = []
                        head = 0
                        #Avoid cuda out of memory. To speed things up, you can query all the points together rather than in batches.
                        while head < position.shape[0]:
                            tail = min(head + 640000, position.shape[0])
                            sum1,color,specular,diffuse = shader(position[head:tail], normal[head:tail], view_direction[head:tail])
                            head += 640000
                            all_sum1.append(sum1)
                            all_color.append(color)
                            all_specular.append(specular)
                            all_diffuse.append(diffuse)
                    elif self.mode == 'test':
                        sum1,color,specular,diffuse = shader(position, normal, view_direction)
            else:
                color = position
                specular = normal
                diffuse = view_direction
                sum1 = position

            if self.mode == 'test' or torch.is_tensor(envmap) or shading==False:
                full[mask_flatten] = color
                dif[mask_flatten] = diffuse
                spe[mask_flatten] = specular
            else:
                full[mask_flatten] = torch.cat(all_color,dim=0)
                dif[mask_flatten] = torch.cat(all_diffuse,dim=0)
                spe[mask_flatten] = torch.cat(all_specular,dim=0)
            normal_img[mask_flatten] = (normal+1)/2
            
            mask = mask.float()
            full = full.view(1, self.h, self.w, 3)
            spe = spe.view(1, self.h, self.w, 3)
            dif = dif.view(1, self.h, self.w, 3)
            normal_img = normal_img.view((1, self.h, self.w, 3))
            mask = dr.antialias(mask, rast, pos, idx, pos_gradient_boost=self.opt.pos_gradient_boost).squeeze(0)
            full = dr.antialias(full, rast, pos, idx, pos_gradient_boost=self.opt.pos_gradient_boost).squeeze(0)
            spe = dr.antialias(spe, rast, pos, idx, pos_gradient_boost=self.opt.pos_gradient_boost).squeeze(0)
            dif = dr.antialias(dif, rast, pos, idx, pos_gradient_boost=self.opt.pos_gradient_boost).squeeze(0)
            normal_img = dr.antialias(normal_img, rast, pos, idx, pos_gradient_boost=self.opt.pos_gradient_boost).squeeze(0)
            full = full  * mask 
            spe = spe *  mask 
            dif = dif * mask 
            normal_img = normal_img * mask

            if self.opt.ssaa > 1:
                mask = scale_img_hwc(mask, (self.h0, self.w0))
                full = scale_img_hwc(full, (self.h0, self.w0))
                spe = scale_img_hwc(spe, (self.h0, self.w0))
                dif = scale_img_hwc(dif, (self.h0, self.w0))
                normal_img = scale_img_hwc(normal_img, (self.h0, self.w0))
            mask = mask.view(*prefix, 1)
            full = full.view(*prefix, 3)
            spe = spe.view(*prefix, 3)
            dif = dif.view(*prefix, 3)
            normal_img = normal_img.view(*prefix, 3)
            full = full + 1 - mask
            spe = spe + 1 - mask
            dif = dif + 1 - mask
            normal_img = normal_img + 1 - mask
            imgs_diff.append(dif)
            imgs_spe.append(spe)
            imgs_full.append(full)
            masks.append(mask)
            normals.append(normal_img)


        return masks,imgs_full,imgs_diff,imgs_spe,normals,torch.tensor(sum1).mean()