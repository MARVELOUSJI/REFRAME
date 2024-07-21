import random
import os
import numpy as np
import torch
import math
from pathlib import Path
import trimesh
from .mesh import Mesh
import nvdiffrast.torch as dr
import xatlas
from scipy.ndimage import binary_dilation, binary_erosion
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import cv2
import json
from torch import nn


def read_mesh(path, device='cpu'):
    mesh_ = trimesh.load_mesh(str(path), process=False)
    vertices = np.array(mesh_.vertices, dtype=np.float32)
    indices = None
    if hasattr(mesh_, 'faces'):
        indices = np.array(mesh_.faces, dtype=np.int32)
    vertex_normals = None
    if hasattr(mesh_, 'vertex_normals'):
        vertex_normals = np.array(mesh_.vertex_normals, dtype=np.float32)
    
    return Mesh(vertices, indices,vertex_normals, device)

def write_mesh(path, mesh):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vertices = mesh.vertices.detach().to('cpu').numpy()
    indices = mesh.indices.detach().to('cpu').numpy() if mesh.indices is not None else None
    mesh_ = trimesh.Trimesh(vertices=vertices, faces=indices,vertex_normals = mesh.vertex_normals.detach().to('cpu').numpy(), process=False)
    mesh_.export(path)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def cor2dir(coordinates):
    coordinates[:,0] *= 90
    coordinates[:,1] *= 180
    z = torch.cos(torch.tensor((coordinates[:,0]+90)*math.pi/180))
    r_xy = (1-z**2)**0.5
    tem = (coordinates[:,1]<0)
    x = r_xy*torch.cos(torch.tensor(coordinates[:,1]*math.pi/180))
    y = (r_xy**2-x**2)**0.5
    y[tem] *= -1
    return torch.stack((x,y,z),dim=1)

def bakeenvmap(rx,ry,shader):
    x = torch.linspace(-1, 1, rx)  
    y = torch.linspace(-1, 1, ry)  
    grid_x, grid_y = torch.meshgrid(x, y)  

    coordinates = torch.stack((grid_x, grid_y), dim=2) 
    coordinates = coordinates.reshape(-1,2)
    dir = cor2dir(coordinates)
    dir_pe = shader.pe_encoder(dir).float()  
    envmap = shader.dir_encoder(dir_pe)
    return envmap.reshape(rx,ry,3)

def dir2polar(dir):
    r =   torch.norm(dir, dim=-1)  
    r_xy = torch.norm(dir[:,:2], dim=-1)
    
    radian_xy = torch.acos(dir[:,-1]/r)
    degree_xy = torch.rad2deg(radian_xy)-90

    radian_z = torch.acos(dir[:,0]/(r_xy+1e-20))
    degree_z = torch.rad2deg(radian_z)

    tem = (dir[:,1]<0)
    degree_z[tem] = -1*degree_z[tem]
    return degree_xy/90.0,degree_z/180.0

#Reference to NeRF2Mesh.
def scale_img_nhwc(x, size, mag='bilinear', min='bilinear'):
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2) 
    if x.shape[1] > size[0] and x.shape[2] > size[1]: 
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else: 
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous() 

def scale_img_hwc(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]

def uvmapping(mesh,opt,shader):
    device = torch.device('cpu')
    if torch.cuda.is_available() and opt.device >= 0:
        device = torch.device(f'cuda:{opt.device}')
    glctx = dr.RasterizeGLContext()
    # glctx = dr.RasterizeCudaContext()
    f = mesh.indices.detach().int()
    v_np = mesh.vertices.detach().cpu().numpy() # [N, 3]
    f_np = f.cpu().numpy() # [M, 3]
    print(f'[INFO] running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}')
    # unwrap uvs
    atlas = xatlas.Atlas()
    atlas.add_mesh(v_np, f_np)
    chart_options = xatlas.ChartOptions()
    chart_options.max_iterations = 0
    pack_options = xatlas.PackOptions()

    atlas.generate(chart_options=chart_options, pack_options=pack_options)
    vmapping, ft_np, vt_np = atlas[0] # [N], [M, 3], [N, 2]

    vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
    ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)

    print(f'[INFO] after running xatlas: v={vt.shape} f={ft.shape}')

    # render uv maps
    uv = vt * 2.0 - 1.0 # uvs to range [-1, 1]
    uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1) # [N, 4]
    h0 = 4096
    w0 = 4096
    if opt.ssaa > 1:
        h = int(h0 * opt.ssaa)
        w = int(w0 * opt.ssaa)
    else:
        h, w = h0, w0

    rast, _ = dr.rasterize(glctx, uv.unsqueeze(0), ft, (h, w)) # [1, h, w, 4]
    xyzs, _ = dr.interpolate(mesh.vertices.detach().unsqueeze(0), rast, f) # [1, h, w, 3]
    mask, _ = dr.interpolate(torch.ones_like(mesh.vertices[:, :1].detach()).unsqueeze(0), rast, f) # [1, h, w, 1]
    normals, _ = dr.interpolate(mesh.vertex_normals.detach().unsqueeze(0), rast, f) # [1, h, w, 3]

    xyzs = xyzs.view(-1, 3)
    normals = normals.view(-1, 3)
    
    eps_ = 1e-6
    normals = torch.tensor(normals)
    normals = normals / torch.sqrt(torch.clamp(torch.sum(normals * normals, -1, keepdim=True),min=eps_))

    normals = (normals + 1 )/(2.0)
    mask = (mask > 0).view(-1)
    feats = torch.zeros(h * w, 6, device=device, dtype=torch.float32)
    feats_normal = torch.zeros(h * w, 3, device=device, dtype=torch.float32)
    with torch.no_grad():
        if mask.any() :
            xyzs = xyzs[mask] 
            all_feats = []
            head = 0
            while head < xyzs.shape[0]:
                tail = min(head + 640000, xyzs.shape[0])
                with torch.cuda.amp.autocast(enabled=True):
                    all_feats.append(shader.hashmapping(xyzs[head:tail]).float())
                head += 640000

            feats[mask] = torch.cat(all_feats, dim=0)



    feats_normal[mask] = normals[mask]
    feats = feats.view(h, w, -1) # 6 channels
    feats_normal = feats_normal.view(h, w, -1) # 3 channels

    mask = mask.view(h, w)
    feats=feats.detach().cpu().numpy()
    feats_normal=feats_normal.detach().cpu().numpy()

    ## NN search as a queer antialiasing ...
    mask = mask.cpu().numpy()
    inpaint_region = binary_dilation(mask, iterations=3)
    inpaint_region[mask] = 0

    search_region = mask.copy()
    not_search_region = binary_erosion(search_region, iterations=2)
    search_region[not_search_region] = 0

    search_coords = np.stack(np.nonzero(search_region), axis=-1)
    inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

    knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_coords)
    _, indices = knn.kneighbors(inpaint_coords)

    feats[tuple(inpaint_coords.T)] = feats[tuple(search_coords[indices[:, 0]].T)]
    feats_normal[tuple(inpaint_coords.T)] = feats_normal[tuple(search_coords[indices[:, 0]].T)]


    if opt.ssaa > 1:
        fe_our = scale_img_hwc(torch.tensor(feats), (h0, w0)).detach().cpu().numpy()
        normal_our = scale_img_hwc(torch.tensor(feats_normal), (h0, w0)).detach().cpu().numpy()
    else:
        fe_our = feats
        normal_our = feats_normal
    feats_diffuse=fe_our[...,:3]

    experiment_dir = opt.output_dir / opt.run_name
    shaders_save_path = experiment_dir / "shaders"

    feats_n=normal_our[...,:]##3
    plt.imsave(os.path.join(shaders_save_path, f'normal_.png'),feats_n) 

    normal_img = cv2.imread(os.path.join(shaders_save_path, f'normal_.png'), cv2.IMREAD_UNCHANGED) 
    normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB)/255.0
    
    re = normal_our-normal_img
    averafe_re = np.log10(1/re.mean()).astype(int)+1
    re = re*10**averafe_re
    re_max = re.max()
    re_min = re.min()
    re = (re-re_min)/(re_max-re_min)
    plt.imsave(shaders_save_path / f'normalre.png', re)

    re_img = cv2.imread(os.path.join(shaders_save_path, f'normalre.png'), cv2.IMREAD_UNCHANGED) 
    re_img = cv2.cvtColor(re_img, cv2.COLOR_BGR2RGB)/255.0
    re_img = re_img*(re_max-re_min)+re_min
    re_img = re_img/10**(averafe_re)
    normal_imgour = normal_img + re_img
    print(f'normal map diff----{(normal_our-normal_imgour).mean()}')
    with open(shaders_save_path /'minmax.txt', 'ab') as f:
        np.savetxt(f, (averafe_re,re_min,re_max))

    feats = (feats * 255).astype(np.uint8)

    feats0 = cv2.cvtColor(feats[..., :3], cv2.COLOR_RGB2BGR) 
    feats1 = cv2.cvtColor(feats[..., 3:], cv2.COLOR_RGB2BGR) 

    if opt.ssaa > 1:
        feats0 = cv2.resize(feats0, (w0, h0), interpolation=cv2.INTER_LINEAR)
        feats1 = cv2.resize(feats1, (w0, h0), interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(os.path.join(shaders_save_path, f'diffusefe_.jpg'), feats0)
    cv2.imwrite(os.path.join(shaders_save_path, f'specularfe_.jpg'), feats1)

    obj_file = os.path.join(shaders_save_path, f'mesh_{0}_.obj')
    mtl_file = os.path.join(shaders_save_path, f'mesh_{0}.mtl')

    print(f'[INFO] writing obj mesh to {obj_file}')
    with open(obj_file, "w") as fp:
        fp.write(f'mtllib mesh_{0}.mtl \n')

        print(f'[INFO] writing vertices {v_np.shape}')
        for v in v_np:
            fp.write(f'v {v[0]} {v[1]} {v[2]} \n')

        print(f'[INFO] writing vertices texture coords {vt_np.shape}')
        for v in vt_np:
            fp.write(f'vt {v[0]} {1 - v[1]} \n') 

        print(f'[INFO] writing faces {f_np.shape}')
        fp.write(f'usemtl defaultMat \n')
        for i in range(len(f_np)):
            fp.write(f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} {f_np[i, 1] + 1}/{ft_np[i, 1] + 1} {f_np[i, 2] + 1}/{ft_np[i, 2] + 1} \n")

    with open(mtl_file, "w") as fp:
        fp.write(f'newmtl defaultMat \n')
        fp.write(f'Ka 1 1 1 \n')
        fp.write(f'Kd 1 1 1 \n')
        fp.write(f'Ks 0 0 0 \n')
        fp.write(f'Tr 1 \n')
        fp.write(f'illum 1 \n')
        fp.write(f'Ns 0 \n')
        fp.write(f'map_Kd diffusefe.jpg \n')      
        # save mlp as json
    params = dict(shader.rgb_net.named_parameters())

    mlp = {}
    for k, p in params.items():
        p_np = p.detach().cpu().numpy().T
        print(f'[INFO] wrting MLP param {k}: {p_np.shape}')
        mlp[k] = p_np.tolist()

    mlp_file = os.path.join(shaders_save_path, f'mlp.json')
    with open(mlp_file, 'w') as fp:
        json.dump(mlp, fp, indent=2)
        
class MLP(nn.Module):  
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True, weight_norm=False):  
        super().__init__()  
        self.dim_in = dim_in  
        self.dim_out = dim_out  
        self.dim_hidden = dim_hidden  
        self.num_layers = num_layers  
  
        net = []  
        for l in range(num_layers):  
  
            in_dim = self.dim_in if l == 0 else self.dim_hidden  
            out_dim = self.dim_out if l == num_layers - 1 else self.dim_hidden  
  
            net.append(nn.Linear(in_dim, out_dim, bias=bias))  
              
              
            if l != self.num_layers - 1:  
                net.append(nn.ReLU(inplace=True))  
  
            if weight_norm:  
                torch.nn.init.normal_(net[l*2].weight, mean=0, std=1e-1)
                if bias: torch.nn.init.constant_(net[l*2].bias, 0.0)

  
        self.net = nn.Sequential(*net)  
  
    def forward(self, x):  
        return self.net(x)   