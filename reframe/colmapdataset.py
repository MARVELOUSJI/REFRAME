import os
import cv2
import tqdm
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import collections
import struct

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) \
                         for camera_model in CAMERA_MODELS])
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)
def read_cameras_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)
def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	# handle exception for the opposite direction input
	if c < -1 + 1e-10:
		return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


class ColmapDataset:
    def __init__(self, opt, device, type='train'):
        super().__init__()
        self.views_per_iter = opt.views_per_iter
        self.current_index = 0
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = 4
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.root_path = opt.datadir 

        self.training = self.type in ['train', 'all', 'trainval']

        # locate colmap dir
        candidate_paths = [
            os.path.join(self.root_path, "colmap_sparse", "0"),
            os.path.join(self.root_path, "sparse", "0"),
            os.path.join(self.root_path, "colmap"),
        ]
        
        self.colmap_path = None
        for path in candidate_paths:
            if os.path.exists(path):
                self.colmap_path = path
                break

        if self.colmap_path is None:
            raise ValueError(f"Cannot find colmap sparse output under {self.root_path}, please run colmap first!")

        camdata = read_cameras_binary(os.path.join(self.colmap_path, 'cameras.bin'))

        self.H = int(round(camdata[1].height / self.downscale))
        self.W = int(round(camdata[1].width / self.downscale))
        print(f'[INFO] ColmapDataset: image H = {self.H}, W = {self.W}')

        imdata = read_images_binary(os.path.join(self.colmap_path, "images.bin"))
        imkeys = np.array(sorted(imdata.keys()))

        img_names = [os.path.basename(imdata[k].name) for k in imkeys]
        for i in range(len(img_names)):  
            img_names[i] = img_names[i].replace('.JPG', '.jpg') 
        img_folder = os.path.join(self.root_path, f"images_{self.downscale}")
        if not os.path.exists(img_folder):
            img_folder = os.path.join(self.root_path, "images")
        img_paths = np.array([os.path.join(img_folder, name) for name in img_names])

        exist_mask = np.array([os.path.exists(f) for f in img_paths])
        print(f'[INFO] {exist_mask.sum()} image exists in all {exist_mask.shape[0]} colmap entries.')
        imkeys = imkeys[exist_mask]
        img_paths = img_paths[exist_mask]

        
        intrinsics = []
        for k in imkeys:
            cam = camdata[imdata[k].camera_id]
            if cam.model in ['SIMPLE_RADIAL', 'SIMPLE_PINHOLE']:
                fl_x = fl_y = cam.params[0] / self.downscale
                cx = cam.params[1] / self.downscale
                cy = cam.params[2] / self.downscale
            elif cam.model in ['PINHOLE', 'OPENCV']:
                fl_x = cam.params[0] / self.downscale
                fl_y = cam.params[1] / self.downscale
                cx = cam.params[2] / self.downscale
                cy = cam.params[3] / self.downscale
            else:
                raise ValueError(f"Unsupported colmap camera model: {cam.model}")
            intrinsics.append(np.array([fl_x, fl_y, cx, cy], dtype=np.float32))
        
        self.intrinsics = torch.from_numpy(np.stack(intrinsics)) 

        poses = []
        for k in imkeys:
            P = np.eye(4, dtype=np.float64)
            P[:3, :3] = imdata[k].qvec2rotmat()
            P[:3, 3] = imdata[k].tvec
            poses.append(P)

        self.poses = np.linalg.inv(np.stack(poses, axis=0)) 
        self.poses[:, :3, 1:3] *= -1
        self.poses = self.poses[:, [1, 0, 2, 3], :]
        self.poses[:, 2] *= -1
        self.poses[:, :3, 3] *= self.scale

 


        all_ids = np.arange(len(img_paths))
        val_ids = all_ids[::8]

        if self.type == 'train':
            train_ids = np.array([i for i in all_ids if i not in val_ids])
            self.poses = self.poses[train_ids]
            self.intrinsics = self.intrinsics[train_ids]
            img_paths = img_paths[train_ids]

        elif self.type == 'test':
            self.poses = self.poses[val_ids]
            self.intrinsics = self.intrinsics[val_ids]
            img_paths = img_paths[val_ids]

        
        self.images = []

        for i, f in enumerate(tqdm.tqdm(img_paths, desc=f'Loading {self.type} data')):

            image = cv2.imread(f, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]

            if image.shape[-1] == 3: 
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255.0
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)/255.0

            if image.shape[0] != self.H or image.shape[1] != self.W:
                image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)

            self.images.append(image)

        self.images = np.stack(self.images, axis=0)


        self.poses = torch.from_numpy(self.poses.astype(np.float32)) # [N, 4, 4]

        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)).float() # [N, H, W, C]
        
        self.near = 0.05
        self.far = 1000 # infinite
        aspect = self.W / self.H

        projections = []
        for intrinsic in self.intrinsics:
            y = self.H / (2.0 * intrinsic[1].item()) 
            projections.append(np.array([[1/(y*aspect), 0, 0, 0], 
                                        [0, -1/y, 0, 0],
                                        [0, 0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)],
                                        [0, 0, -1, 0]], dtype=np.float32))
        self.projections = torch.from_numpy(np.stack(projections)) 
        self.mvps = self.projections @ torch.inverse(self.poses)
        self.index_buffer = list(range(len(self.images)))

        self.intrinsics = self.intrinsics.to(self.device)
        self.poses = self.poses.to(self.device)
        if self.images is not None:
            self.images = self.images.to(self.device)
        self.mvps = self.mvps.to(self.device)


    def collate(self, index):
        results = {'H': self.H, 'W': self.W}
        if self.opt.rendermode != 'volume':
            ourindex = []
            images = []
            mvps = []
            camera_location = []
            if self.training:                           
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
                ourindex = index[0]
                images.append(self.images[ourindex])
                ourindex.append(ourindex)
                mvps.append(self.mvps[ourindex])
                camera_location.append(self.poses[ourindex][:3,3])
                
            results['images'] = images
            results['index'] = ourindex
            results['mvp'] = mvps
            results['camera_location'] = camera_location
                        
        return results
    def dataloader(self):
        size = len(self.poses)
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self 
        loader.has_gt = self.images is not None
        return loader
