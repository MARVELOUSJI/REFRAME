import os
import cv2
import json
import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import random
from copy import deepcopy

def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    pose[:3, 3] = pose[:3, 3] * scale + np.array(offset)
    pose = pose.astype(np.float32)
    return pose

class BlenderDataset:
    modes = ['random', 'sequential', 'sequential_shuffled']
    def __init__(self, opt, device, epsilon, type='train', mode = 'sequential_shuffled',vertices=None):
        super().__init__()
        if not mode in self.modes:
            raise ValueError(f"Unknown mode '{mode}'. Available modes are {', '.join(self.modes)}")
        self.mode = mode
        self.views_per_iter = opt.views_per_iter
        self.current_index = 0
        
        self.opt = opt
        self.device = device
        self.type = type 
        self.root_path = opt.datadir
        self.scale = opt.scale 
        self.bound = opt.bound         
        with open(os.path.join(self.root_path, f'transforms_{type}.json'), 'r') as f:
            transform = json.load(f)

        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) 
            self.W = int(transform['w']) 
        else:
            self.H = self.W = None
        scale_mat = None
        frames = np.array(transform["frames"])
        if opt.refneus and isinstance(vertices, np.ndarray):
            bbox_max = np.max(vertices, axis=0) 
            bbox_min = np.min(vertices, axis=0) 
            center = (bbox_max + bbox_min) * 0.5
            radius = np.linalg.norm(vertices - center, ord=2, axis=-1).max() 
            scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
            scale_mat[:3, 3] = center
            self.scale_mat = deepcopy(scale_mat)
            self.scale_mat[0, 0] *= 150
            self.scale_mat[1, 1] *= 150
            self.scale_mat[2, 2] *= 150
            self.scale_mat[:3, 3] *= 150
        
        self.poses = []
        self.images = []
        self.camera_R = []
        self.camera_T = []

        for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
            f_path = os.path.join(self.root_path, f['file_path'])
            if '.' not in os.path.basename(f_path):
                f_path += '.png' 

            if not os.path.exists(f_path):
                print(f'[WARN] {f_path} not exists!')
                continue
    
            pose = np.array(f['transform_matrix'], dtype=np.float32) 
           
            if isinstance(scale_mat, np.ndarray):
                pose[:, 3:] = (np.linalg.inv(scale_mat)) @ pose[:, 3:]
            else:
                pose = nerf_matrix_to_ngp(pose, scale=self.scale)
            location = np.asarray(pose)[:3,3]
            rotation = np.asarray(pose)[:3,:3]
            new_rotation = rotation @ np.array([[1,0,0],
                                                [0,-1,0],
                                                [0,0,-1]])
            
            w2c_rotation = np.linalg.inv(new_rotation).astype(np.float32) 
            w2c_location = (w2c_rotation @ (- location)).astype(np.float32)
            
            image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
            if self.H is None or self.W is None:
                self.H = image.shape[0] 
                self.W = image.shape[1] 
            if image.shape[-1] == 3: 
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255.0
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)/255.0
            if image.shape[0] != self.H or image.shape[1] != self.W:
                image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)

            self.poses.append(pose)
            self.images.append(image)
            self.camera_R.append(w2c_rotation)
            self.camera_T.append(w2c_location)
        
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0)).to(self.device) 
        self.camera_R = torch.from_numpy(np.stack(self.camera_R, axis=0)).to(self.device) 
        self.camera_T = torch.from_numpy(np.stack(self.camera_T, axis=0)).to(self.device) 
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)).float().to(self.device) 
        self.index_buffer = list(range(len(self.images)))

        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) 
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) 
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] ) if 'cx' in transform else (self.W / 2.0)
        cy = (transform['cy'] ) if 'cy' in transform else (self.H / 2.0)
    
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])
        self.camera_K = torch.from_numpy(np.array([[fl_x,0,0.5*self.W],
              [0,fl_y,0.5*self.H],
              [0,0,1]]).astype(np.float32)).to(self.device)

        aabbconer = torch.FloatTensor([
            [-opt.bound, -opt.bound, -opt.bound],
            [opt.bound, -opt.bound, -opt.bound],
            [opt.bound, opt.bound, -opt.bound],
            [-opt.bound, opt.bound, -opt.bound],
            [-opt.bound, -opt.bound, opt.bound],
            [opt.bound, -opt.bound, opt.bound],
            [opt.bound, opt.bound, opt.bound],
            [-opt.bound, opt.bound, opt.bound]
        ]).to(self.device)
        self.set_near_far(aabbconer, epsilon)
    
    def collate(self, index):
        results = {'H': self.H, 'W': self.W}
        
        ourindex = []
        images = []
        mvps = []
        camera_location = []
        if self.type=='train':
            if self.mode == 'random':
                ourindex = np.random.choice(len(self.images), self.views_per_iter, replace=False)
                for i in range(self.views_per_iter):
                    images.append(self.images[ourindex[i]])
                    mvps.append(self.mvps[ourindex[i]])
                    camera_location.append(self.poses[ourindex[i]][:3,3])
                
            elif self.mode == 'sequential':
                for _ in range(self.views_per_iter):
                    self.current_index = (self.current_index + 1) % len(self.images)
                    ourindex.append(self.current_index)
                    images.append(self.images[self.current_index])
                    mvps.append(self.mvps[self.current_index])
                    camera_location.append(self.poses[self.current_index][:3,3])
                    

            elif self.mode == 'sequential_shuffled':
                for _ in range(self.views_per_iter):
                    view_index = self.index_buffer[self.current_index]
                    ourindex.append(view_index)
                    images.append(self.images[view_index])
                    mvps.append(self.mvps[view_index])
                    camera_location.append(self.poses[view_index][:3,3])
                    
                    
                    self.current_index = (self.current_index + 1) 
                    if self.current_index >= len(self.images):
                        random.shuffle(self.index_buffer)
                        self.current_index = 0
        
        else:
            index = index[0]
            images.append(self.images[index])
            ourindex.append(index)
            mvps.append(self.mvps[index])
            camera_location.append(self.poses[index][:3,3])
            
        results['images'] = images
        results['index'] = ourindex
        results['mvp'] = mvps
        results['camera_location'] = camera_location

        return results

    def dataloader(self):
        size = len(self.poses)
        if self.type == 'train':
            shuffle=True
        else:
            shuffle=False
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=shuffle, num_workers=0)
        loader._data = self 
        loader.has_gt = self.images is not None
        return loader
    
    def project(self, points,index, depth_as_distance=False):

        points_c = points @ torch.transpose(self.camera_R[index], 0, 1) + self.camera_T[index]
        pixels = points_c @ torch.transpose(self.camera_K, 0, 1)
        pixels = pixels[..., :2] / pixels[..., 2:]
        depths = points_c[..., 2:] if not depth_as_distance else torch.norm(points_c, p=2, dim=-1, keepdim=True)
        return torch.cat([pixels, depths], dim=-1)

    def set_near_far(self, samples, epsilon=0.1):

        mins = []
        maxs = []
        for i in range(len(self.images)):
            samples_projected = self.project(samples,i, depth_as_distance=True)
            mins.append(samples_projected[...,2].min())
            maxs.append(samples_projected[...,2].max())

        near, far = min(mins), max(maxs)
        self.near = near - (near * epsilon)
        self.far = far + (far * epsilon)
        print(f'near{self.near}-----far{self.far}')
        self.projection = torch.tensor([[2.0*self.intrinsics[0]/self.W,           0,       1.0 - 2.0 * self.intrinsics[2] / self.W,                                                0],
                                        [         0,  -2.0*self.intrinsics[1]/self.H,       1.0 - 2.0 * self.intrinsics[3] / self.H,                                                0],
                                        [         0,                0,                  -(self.far+self.near)/(self.far-self.near),     -(2*self.far*self.near)/(self.far-self.near)],
                                        [         0,                0,                               -1,                                                                        0.0]], device=self.device).float()
        self.mvps = self.projection.unsqueeze(0) @ torch.inverse(self.poses).to(self.device)

   
