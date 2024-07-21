from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import torch
from tqdm import tqdm
import tensorboardX
import os
import matplotlib.pyplot as plt
from shutil import copyfile
import imageio
from pytorch_msssim import SSIM
import lpips
import cv2
import json
import trimesh
from pathlib import Path

from reframe import (
    Mesh, SurfaceRenderer,BlenderDataset,ColmapDataset,velearner, tcnnshader,norlearner,utils
)


to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

if __name__ == '__main__':
    parser = ArgumentParser(description='REFRAME: Reflective Surface Real-Time Rendering for Mobile Devices', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datadir', type=Path, default="./data/lego/", help="Path to the data directory")
    parser.add_argument('--outputdir', type=Path, default="./output/", help="Path to the output directory")
    parser.add_argument('--initial_mesh', type=str, default="./premesh/lego.obj", help="Path to the initial coarse mesh")
    parser.add_argument('--epoch', type=int, default=250, help="Total number of epoch")
    parser.add_argument('--run_name', type=str, default='lego', help="Name of this run")
    parser.add_argument('--scale', type=float, default=0.8, help="Scale of camera loaction")
    parser.add_argument('--foreground', type=float, default=0.3, help="Foregroung bound of the scene, need to adjust deponds on scenes")
    parser.add_argument('--bound', type=float, default=1.0, help="Bound of the scene, 1 for object")
    parser.add_argument('--lr_geo', type=float, default=1e-3, help="Learning rate for the geometry learner")
    parser.add_argument('--lr_shader', type=float, default=1e-3, help="Learning rate for the shader")
    parser.add_argument('--lr_shader_step', type=float, default=0.95, help="Step size for the learning rate for the shader")    
    parser.add_argument('--lr_geo_step', type=float, default=0.95, help="Step size for the learning rate for the geometry learner")    
    parser.add_argument('--lr_frequency_sha', type=int, default=2, help="Frequency to update the shader lr")
    parser.add_argument('--lr_frequency_geo', type=int, default=2, help="Frequency to update the geometry learner lr")
    parser.add_argument('--lr_fre_sha_step', type=float, default=2.0, help="Shader lr frequency step")
    parser.add_argument('--lr_fre_geo_step', type=float, default=2.0, help="Geometry learner lr frequency step")
    parser.add_argument('--lr_scheduler_geo', type=str, default='cos', help="Lr schedule for the geometry module")
    parser.add_argument('--lr_scheduler_sh', type=str, default='cos', help="Lr schedule for the shader")
    parser.add_argument('--save_frequency', type=int, default=250, help="Frequency of mesh and shader saving (in epoch)")
    parser.add_argument('--visualization_frequency', type=int, default=250, help="Frequency of visualization (in epoch)")
    parser.add_argument('--device', type=int, default=0, choices=([-1] + list(range(torch.cuda.device_count()))), help="GPU to use; -1 is CPU")
    parser.add_argument('--weight_mask', type=float, default=100, help="Weight of the mask term")
    parser.add_argument('--weight_ssim', type=float, default=3, help="Weight of the ssim term")
    parser.add_argument('--weight_normal', type=float, default=0.1, help="Weight of the normal term")
    parser.add_argument('--weight_shading', type=float, default=1, help="Weight of the shading term")
    parser.add_argument('--weight_diffuse', type=float, default=0.001, help="Weight of the diffuse term")
    parser.add_argument('--weight_max1', type=float, default=0.00001, help="Weight of the max1 term")
    parser.add_argument('--shading_percentage', type=float, default=1, help="Percentage of valid pixels(0-1)")
    parser.add_argument('--shader_path', type=str, default=None, help="Path for the pretrained shader")
    parser.add_argument('--envmap_path', type=str, default=None, help="Path for the pretrained envmap")
    parser.add_argument('--test', type=int, default=0, help="Whether do testing")
    parser.add_argument('--L', type=int, default=16, help="Hash table layer number")
    parser.add_argument('--mlpoff', type=int, default=1, help="Whether to learn the offset by a network")
    parser.add_argument('--pos_gradient_boost', type=float, default=1, help="Nvdiffrast option")
    parser.add_argument('--ssaa', type=int, default=2, help="Super sampling rate")
    parser.add_argument('--uvmap', type=int, default=0, help="Whether to perform uvmap")
    parser.add_argument('--views_per_iter', type=int, default=1, help="Number of views used per iteration.")
    parser.add_argument('--refneus', type=int, default=1,help="Whether using refneus's mesh for initialization")
    parser.add_argument('--resolutionx', type=int, default=360,help="Resolution for environment feature map")
    parser.add_argument('--resolutiony', type=int, default=720,help="Resolution for environment feature map")
    parser.add_argument('--dataset',  type=str, default='blender',help="Dataset type")
    parser.add_argument('--region',  type=int, default=1,help="1 for object, 2 for open scene")
    parser.add_argument('--difgeo',  type=int, default=0,help="If region is 2, whether use different geometry learner for foreground and background.")
    parser.add_argument('--wenvlearner',  type=int, default=1,help="Get environment map through environment learner or direct optimization.")
    
    args = parser.parse_args()
    utils.seed_everything(0)
    torch.autograd.set_detect_anomaly(True)

    device = torch.device('cpu')
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    print(f"Using device {device}")

    # Create directories.
    run_name = args.run_name if args.run_name is not None else args.datadir.parent.name
    experiment_dir = args.outputdir / run_name

    images_path = experiment_dir / "images"
    images_path.mkdir(parents=True, exist_ok=True)
    
    images_diffuse_path = experiment_dir / "images_diffuse"
    images_diffuse_path.mkdir(parents=True, exist_ok=True)

    images_specular_path = experiment_dir / "images_specular"
    images_specular_path.mkdir(parents=True, exist_ok=True)

    normal_img_path = experiment_dir / "images_normal"
    normal_img_path.mkdir(parents=True, exist_ok=True)
    
    meshes_path = experiment_dir / "meshes"
    meshes_path.mkdir(parents=True, exist_ok=True)

    shaders_save_path = experiment_dir / "shaders"
    shaders_save_path.mkdir(parents=True, exist_ok=True)

    code_path = experiment_dir/ "code"
    os.makedirs(os.path.join(code_path, 'nds'), exist_ok=True)

    writer = tensorboardX.SummaryWriter(os.path.join(experiment_dir, 'tensorboard'))

    #Code back up.
    dir_lis = ['./reframe']
    for dir_name in dir_lis:
        cur_dir = os.path.join(code_path, dir_name)
        os.makedirs(cur_dir, exist_ok=True)
        files = os.listdir(dir_name)
        for f_name in files:
            if f_name[-3:] == '.py':
                copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))
    copyfile('main.py', os.path.join(code_path, 'main.py'))
    
    # Save args for this execution
    with open(experiment_dir / "args.txt", "w") as text_file:
        print(f"{args}", file=text_file)


    mesh_initial: Mesh = None    
    mesh_initial = utils.read_mesh(args.initial_mesh, device=device)
    hash_bound = [args.bound]
    #Define network. If not train (test or uvmap) then skip.
    if not (args.test or args.uvmap):
        vertices = None
        if args.refneus: 
            mesh_ = trimesh.load_mesh(f'{args.datadir}/points_of_interest.ply', process=False)
            vertices = np.array(mesh_.vertices, dtype=np.float32)
        #object level 
        if args.dataset=='blender':
            train_dataset = BlenderDataset(args, device=device, epsilon=0.5,type='train',vertices = vertices)
            hash_bound = [args.bound]
        #scene level
        elif args.dataset=='colmap':
            args.weight_mask = 0
            args.region = 2
            args.ssaa = 1
            hash_bound = [args.foreground,args.bound]
            train_dataset = ColmapDataset(args, device=device,type='train')
            if args.difgeo:
                #Indices for foreground and background vertex. With them you can optmize mesh separately.
                indices = []
                x_min, x_max = torch.tensor(-args.bound), torch.tensor(args.bound)
                y_min, y_max = torch.tensor(-args.bound), torch.tensor(args.bound)
                z_min, z_max = torch.tensor(-args.bound), torch.tensor(args.bound)
                indices_mask = (mesh_initial.vertices[:,0] > x_min) & (mesh_initial.vertices[:,0] < x_max)& (mesh_initial.vertices[:,1] > y_min)& (mesh_initial.vertices[:,1] < y_max)& (mesh_initial.vertices[:,2] > z_min)& (mesh_initial.vertices[:,2] < z_max)
                indices.append(torch.nonzero(indices_mask, as_tuple=False).squeeze())  
                indices.append(torch.nonzero(~indices_mask, as_tuple=False).squeeze())  
                
        train_loader = train_dataset.dataloader()
        
        
        if args.refneus:
            mesh_initial.vertices = mesh_initial.vertices @ torch.tensor(np.linalg.inv(train_dataset.scale_mat)[:3, :3].T).cuda() + torch.tensor(np.linalg.inv(train_dataset.scale_mat)[:3, 3][np.newaxis, :]).cuda()
        
        if args.mlpoff:
            ve_learner = []
            ve_parameters = []
            for i in range(args.region):
                bou=hash_bound[i]
                ve_learner.append(velearner(bound=bou,device= device,opt = args))
                ve_parameters += ve_learner[i].parameters()
            optimizer_vertices=torch.optim.Adam(ve_parameters, lr=args.lr_geo)
        else:
            vertex_offsets = torch.zeros_like(mesh_initial.vertices)
            vertex_offsets.requires_grad = True
            optimizer_vertices = torch.optim.Adam([vertex_offsets], lr=args.lr_geo)
        
        nor_parameters = []
        nor_learner = []
        for i in range(args.region):
            bou=hash_bound[i]    
            nor_learner.append(norlearner(bound=bou,device= device,opt = args))
            nor_parameters += nor_learner[i].parameters()
        optimizer_normal=torch.optim.Adam(nor_parameters, lr=args.lr_geo)
        
        renderer = SurfaceRenderer(device=device,opt=args,h0 = train_dataset.H,w0=train_dataset.W,mode='train')
        if args.lr_scheduler_geo == 'step':
            sche_vertex = torch.optim.lr_scheduler.StepLR(optimizer_vertices, step_size=args.lr_frequency_geo, gamma=args.lr_geo_step)
            sche_normal = torch.optim.lr_scheduler.StepLR(optimizer_normal, step_size=args.lr_frequency_geo, gamma=args.lr_geo_step)
            
        elif args.lr_scheduler_geo == 'cos':
            sche_vertex = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_vertices, args.lr_frequency_geo, T_mult=int(args.lr_fre_geo_step), eta_min=1e-6)
            sche_normal = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_normal,  args.lr_frequency_geo, T_mult=int(args.lr_fre_geo_step), eta_min=1e-6)

        elif args.lr_scheduler_geo == 'reducelronpla':
            sche_vertex = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_vertices, mode='min', factor=args.lr_geo_step, patience=args.lr_frequency_geo, threshold=0.01, threshold_mode='rel', min_lr=5e-5)
            sche_normal = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_normal, mode='min', factor=args.lr_geo_step, patience=args.lr_frequency_geo, threshold=0.01, threshold_mode='rel', min_lr=5e-5)
        
        loss_weights = {
            "mask": args.weight_mask,
            "normal": args.weight_normal,
            "shading": args.weight_shading,
            "ssim": args.weight_ssim,
            "diffuse":args.weight_diffuse,
            "max1":args.weight_max1,
        }   
        losses = {k: torch.tensor(0.0, device=device) for k in loss_weights}    
    envir_map = None
    if args.wenvlearner==0:
        # envmap = torch.randn([args.resolutionx,args.resolutiony,3]).cuda()
        #We find better initialization for envmap will lead to better performance.
        envir_map = 0.4 + 0.2 * torch.rand([args.resolutionx,args.resolutiony,3]).cuda()
        envir_map.requires_grad = True
        optimizer_envmap = torch.optim.Adam([envir_map], lr=1e-3)
        sche_envmap = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_envmap, args.lr_frequency_sha, T_mult=int(args.lr_fre_sha_step), eta_min=1e-6)
    shader = tcnnshader(bound=hash_bound[-1],device= device,opt = args)
    optimizer_shader = torch.optim.Adam(shader.parameters(), lr=args.lr_shader)
    
    if args.shader_path:
        shader = shader.load(args.shader_path,device)
    if args.envmap_path:
        envir_map = torch.load(args.envmap_path, map_location=device) 

    if args.lr_scheduler_sh == 'step':
        sche_shader = torch.optim.lr_scheduler.StepLR(optimizer_shader, step_size=args.lr_frequency_sha, gamma=args.lr_shader_step)
    elif args.lr_scheduler_sh == 'cos':
        sche_shader = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_shader, args.lr_frequency_sha, T_mult=int(args.lr_fre_sha_step), eta_min=1e-4)
    elif args.lr_scheduler_sh == 'reducelronpla':
        sche_shader = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_shader, mode='min', factor=args.lr_shader_step, patience=args.lr_frequency_sha, threshold=0.1, threshold_mode='rel', min_lr=6e-4)
    
    
    progress_bar = tqdm(range(1, args.epoch + 1))
    MSE_function = torch.nn.MSELoss(reduction='none')
    SSIM_function = SSIM(data_range=1, size_average=True)
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    iteration = 0
    for epoch in progress_bar:
        if args.test or args.uvmap:
            break    
        progress_bar.set_description(desc=f'epoch {epoch}') 
        for data in train_loader:
            smooth_weight = torch.clamp(torch.tensor(epoch)/100.0,0.0,1.0)
            if args.mlpoff:
                if args.region==2 and args.difgeo:
                    vertex_offsets = torch.zeros_like(mesh_initial.vertices)
                    vertex_offsets[indices[0]] = ve_learner[0](mesh_initial.vertices[indices[0]])
                    vertex_offsets[indices[1]] = ve_learner[1](mesh_initial.vertices[indices[1]])
                else:
                    vertex_offsets = ve_learner[-1](mesh_initial.vertices)
            
            if args.region==2 and args.difgeo:
                normal_offsets = torch.zeros_like(mesh_initial.vertex_normals)
                normal_offsets[indices[0]] = nor_learner[0](mesh_initial.vertices[indices[0]],mesh_initial.vertex_normals[indices[0]])
                normal_offsets[indices[1]] = nor_learner[1](mesh_initial.vertices[indices[1]],mesh_initial.vertex_normals[indices[1]])
            
            else:     
                normal_offsets =  nor_learner[-1](mesh_initial.vertices,mesh_initial.vertex_normals)

            # mesh = mesh_initial.with_vertices(mesh_initial.vertices + smooth_weight*vertex_offsets,mesh_initial.vertex_normals)
            mesh = mesh_initial.with_vertices(mesh_initial.vertices + smooth_weight*vertex_offsets)
            mesh.vertex_normals += normal_offsets*smooth_weight
            # mesh.vertex_normals = mesh.vertex_normals.contiguous()
            if args.wenvlearner:
                masks,imgs_full,imgs_diff,imgs_spe,normals,sum1 = renderer.render(train_dataset,data, mesh, channels=['mask', 'position', 'normal'], shader=shader,envmap=None) 
                
            else:
                masks,imgs_full,imgs_diff,imgs_spe,normals,sum1 = renderer.render(train_dataset,data, mesh, channels=['mask', 'position', 'normal'], shader=shader,envmap=envir_map) 
            
            
            
            if loss_weights['mask'] > 0:
                loss_tmp = 0.0
                for i, mask in enumerate(masks):
                    loss_tmp += (MSE_function(data['images'][i][..., 3:], mask)).mean()
                losses['mask'] = loss_tmp / args.views_per_iter
                print(f"loss_mask:{losses['mask']}--iteration:{iteration}")

            if loss_weights['normal'] > 0:
                if args.region==1 or args.difgeo==0: 
                    losses['normal'] = torch.abs(normal_offsets).mean()
                else:
                    losses['normal'] = torch.abs(normal_offsets[indices[0]]).mean()
                print(f"loss_normal:{losses['normal']}--iteration:{iteration}")

            if loss_weights['shading'] > 0:
                shading_tmp = 0.0
                diffuse_tmp = 0.0
                max1_tmp = 0.0
                psnr = 0.0
                ssim_tem = 0.0
                for i, img in enumerate(imgs_full):
                    if data['images'][i].shape[-1]==4:
                        target = data['images'][i][..., :3] * data['images'][i][..., 3:] + 1 - data['images'][i][..., 3:]
                    elif data['images'][i].shape[-1]==3:
                        target  = data['images'][i]
                    loss_psnr = MSE_function(target.view(-1),img.view(-1)).mean() 
                    psnr += -10 * torch.log10(loss_psnr)
                    mask_ra = (masks[i] > 0).squeeze()
                    target_mask = target[mask_ra]
                    img_mask = img[mask_ra]
                    diffuse_mask = imgs_diff[i][mask_ra]
                    shading_tmp += MSE_function(target_mask,img_mask).mean()
                    diffuse_tmp += MSE_function(target_mask,diffuse_mask).mean()
                    max1_tmp += sum1
                    ssim_tem += 1-SSIM_function(img.permute(2,0,1).unsqueeze(0), target.permute(2,0,1).unsqueeze(0))

                losses['ssim'] = ssim_tem / args.views_per_iter
                losses['shading'] = shading_tmp / args.views_per_iter
                losses['diffuse'] = diffuse_tmp / args.views_per_iter
                losses['max1'] = max1_tmp / args.views_per_iter
                psnr = psnr/ args.views_per_iter
                print(f"loss_shading:{losses['shading']}--psnr:{psnr}--iteration:{iteration}")
            loss = torch.tensor(0., device=device)
            for k, v in losses.items():
                loss += v * loss_weights[k]
            
            print(f"loss_total:{loss}--iteration:{iteration}")
            writer.add_scalar(f'Train/PSNR', psnr.item(), iteration)
            writer.add_scalar(f'Train/loss_total', loss.item(), iteration)
            writer.add_scalar(f'Train/loss_mask', losses['mask'].item(), iteration)
            writer.add_scalar(f'Train/loss_normal', losses['normal'].item(), iteration)
            writer.add_scalar(f'Train/loss_max1', losses['max1'].item(), iteration)
            writer.add_scalar(f'Train/loss_diffuse', losses['diffuse'].item(), iteration)
            writer.add_scalar(f'Train/loss_shading', losses['shading'].item(), iteration)
            writer.add_scalar(f'Train/ssim', losses['ssim'].item(), iteration)
            writer.add_scalar(f'Train/vertex(max)', vertex_offsets.max(), iteration)
            writer.add_scalar(f'Train/vertex(min)', vertex_offsets.min(), iteration)
            writer.add_scalar('Train/lr_geo', optimizer_vertices.param_groups[0]['lr'], iteration)
            writer.add_scalar('Train/lr_shader', optimizer_shader.param_groups[0]['lr'], iteration)
            
            optimizer_vertices.zero_grad()
            optimizer_shader.zero_grad()
            optimizer_normal.zero_grad()
            if args.wenvlearner==0:
                optimizer_envmap.zero_grad()
            loss.backward()
            optimizer_vertices.step()
            optimizer_shader.step()
            optimizer_normal.step()
            if args.wenvlearner==0:
                optimizer_envmap.step()
                
            progress_bar.set_postfix({'loss': loss.detach().cpu()})
            
            # Visualizations
            if (args.visualization_frequency > 0)  and (epoch == 1 or epoch % args.visualization_frequency == 0 or epoch == args.epoch):
                with torch.no_grad():
                    for i, img in enumerate(imgs_full):
                        shaded_path = (images_path /'train'/f'{epoch}') 
                        shaded_path.mkdir(parents=True, exist_ok=True)
                        shaded_image_train = torch.clamp(img, 0, 1)  
                        viewindex = data['index'][i]
                        plt.imsave(shaded_path / f'view{str(viewindex)}.png', shaded_image_train.cpu().numpy())
                        
                        shaded_path = (images_diffuse_path /'train'/f'{epoch}') 
                        shaded_path.mkdir(parents=True, exist_ok=True)
                        shaded_image_train = torch.clamp(imgs_diff[i], 0, 1)  
                        plt.imsave(shaded_path / f'view{str(viewindex)}.png', shaded_image_train.cpu().numpy())
                        
                        shaded_path = (images_specular_path /'train'/f'{epoch}') 
                        shaded_path.mkdir(parents=True, exist_ok=True)
                        shaded_image_train = torch.clamp(imgs_spe[i], 0, 1)  
                        plt.imsave(shaded_path / f'view{str(viewindex)}.png', shaded_image_train.cpu().numpy())
                        
                        normal_path = (normal_img_path/'train'/f'{epoch}')
                        normal_path.mkdir(parents=True, exist_ok=True)
                        normal_img = torch.clamp(normals[i],0,1)
                        plt.imsave(normal_path / f'view{str(viewindex)}.png', normal_img.cpu().numpy())
                        
                            
                
            iteration = iteration + 1
       
        if args.lr_scheduler_sh == 'reducelronpla':
            sche_shader.step(losses['shading'])
        else:
            sche_shader.step()
        if args.lr_scheduler_geo == 'reducelronpla':
            sche_vertex.step(losses['mask'])
            sche_normal.step(losses['normal'])
        else:
            sche_vertex.step()
            sche_normal.step()                    
        if args.wenvlearner==0:
            sche_envmap.step()
        if (args.save_frequency > 0) and (epoch == 1 or epoch % args.save_frequency == 0 or epoch == args.epoch):
            with torch.no_grad():
                shader.save(shaders_save_path / f'shader_{epoch:06d}.pt')
                if args.wenvlearner==0:
                    torch.save(envir_map,shaders_save_path / f'envmap_{epoch:06d}.pt')
                if args.refneus:
                    mesh.vertices = mesh.vertices @ torch.tensor(train_dataset.scale_mat[:3, :3].T).cuda() + torch.tensor(train_dataset.scale_mat[:3, 3][np.newaxis, :]).cuda()
                utils.write_mesh(meshes_path / f"mesh_{epoch:06d}.obj", mesh)     
               
  
                    
    if epoch == args.epoch:
        args.test = 1
        mesh_initial = mesh
        args.uvmap = 1

    if args.test:
       with torch.no_grad():
            vertices = None
            if args.refneus:
                mesh_ = trimesh.load_mesh(f'{args.datadir}/points_of_interest.ply', process=False)
                vertices = np.array(mesh_.vertices, dtype=np.float32)
            if args.dataset=='blender':
                test_dataset = BlenderDataset(args, device=device, epsilon=0.5,type='test',vertices = vertices)
            elif args.dataset=='colmap':
                test_dataset = ColmapDataset(args, device=device,type='test')
            
            test_loader = test_dataset.dataloader()
            renderer = SurfaceRenderer(device=device,opt=args,h0 = test_dataset.H,w0=test_dataset.W,mode='test')
            if args.refneus:
                mesh_initial.vertices = mesh_initial.vertices @ torch.tensor(np.linalg.inv(test_dataset.scale_mat)[:3, :3].T).cuda() + torch.tensor(np.linalg.inv(test_dataset.scale_mat)[:3, 3][np.newaxis, :]).cuda()
            
            if args.wenvlearner:
                env_map  = utils.bakeenvmap(args.resolutionx,args.resolutiony,shader)                
            else:
                env_map = envir_map
            
            with open(shaders_save_path / 'envmap.json', 'w') as f:
                json.dump(env_map.tolist(), f)
            
            envmapmax = env_map.max()
            envmapmin = env_map.min()
            env_map_img = (env_map - envmapmin)/(envmapmax - envmapmin)
            plt.imsave(shaders_save_path / f'envmap.png', env_map_img.cpu().numpy())
  
            env_map_img = cv2.imread(os.path.join(shaders_save_path, f'envmap.png'), cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
            env_map_img = cv2.cvtColor(env_map_img, cv2.COLOR_BGR2RGB)/255.0
            env_map_img = torch.tensor(env_map_img).cuda()*(envmapmax - envmapmin)+envmapmin
            
            re = env_map-env_map_img
            averafe_re = torch.log10(1/re.mean()).int()+1
            re = re*10**averafe_re
            re_max = re.max()
            re_min = re.min()
            re = (re-re_min)/(re_max-re_min)
            plt.imsave(shaders_save_path / f'reimg.png', re.cpu().numpy())
            
            re_img = cv2.imread(os.path.join(shaders_save_path, f'reimg.png'), cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
            re_img = cv2.cvtColor(re_img, cv2.COLOR_BGR2RGB)/255.0
            re_img = torch.tensor(re_img).cuda()*(re_max-re_min)+re_min
            re_img = re_img/10**(averafe_re)
            env_mapour = env_map_img + re_img
            print(f'env map diff----{(env_map-env_mapour).mean()}')
            np.savetxt(shaders_save_path /'minmax.txt', (envmapmin.cpu().numpy(), envmapmax.cpu().numpy(),averafe_re.cpu().numpy(),re_min.cpu().numpy(),re_max.cpu().numpy()))  

            view_count=0
            
            psnr = 0
            psnr_diff = 0
            psnr_envmapour = 0
            
            ssim_full = 0
            ssim_diff = 0
            ssim_envmapour = 0

            lpip_vgg=torch.tensor(0).cuda()
            lpip_vgg_diff=torch.tensor(0).cuda()
            lpip_vgg_envmapour=torch.tensor(0).cuda()

            
            shaded_path = (images_path /'test') 
            shaded_path_diff = (images_diffuse_path / 'test') 
            shaded_path_spe = (images_specular_path / 'test') 
            shaded_path_envmapour = (images_path /'envmapour') 
            
            shaded_path.mkdir(parents=True, exist_ok=True) 
            shaded_path_diff.mkdir(parents=True, exist_ok=True)
            shaded_path_spe.mkdir(parents=True, exist_ok=True)
            shaded_path_envmapour.mkdir(parents=True, exist_ok=True)
            
            imgs = []
            dif = []
            spe = []
            imgs_envmapour = []

            for data in test_loader:                           
                if args.wenvlearner:
                    masks,imgs_full,imgs_diff,imgs_spe,normals,sum1 = renderer.render(test_dataset,data, mesh_initial, channels=['mask', 'position', 'normal'], shader=shader,envmap=None) 
                else:
                    masks,imgs_full,imgs_diff,imgs_spe,normals,sum1 = renderer.render(test_dataset,data, mesh_initial, channels=['mask', 'position', 'normal'], shader=shader,envmap=env_mapour) 
                
                if data['images'][0].shape[-1]==4:
                    gt = data['images'][0][...,:3] * data['images'][0][...,3:] + 1 - data['images'][0][...,3:]
                else:
                    gt = data['images'][0]
                shaded_image = torch.clamp(imgs_full[0], 0, 1)   
                mseloss = MSE_function(shaded_image.reshape(-1,3), gt.reshape(-1,3)).mean()
                psnr_tem = -10 * torch.log10(mseloss)
                psnr += psnr_tem
                ssim_tem = SSIM_function(gt.permute(2,0,1).unsqueeze(0), shaded_image.permute(2,0,1).unsqueeze(0))
                ssim_full = ssim_full+ ssim_tem
                lpip_tem = loss_fn_vgg(gt.permute(2,0,1), shaded_image.permute(2,0,1))
                lpip_vgg = lpip_vgg + lpip_tem
                print(f'Testing psnr:{psnr_tem}--ssim:{ssim_tem}--lpip_vgg{lpip_tem} for view{str(view_count)}')
                
                shaded_image_diff = torch.clamp(imgs_diff[0], 0, 1)   
                mseloss = MSE_function(shaded_image_diff.reshape(-1,3), gt.reshape(-1,3)).mean()
                psnr_tem = -10 * torch.log10(mseloss)
                psnr_diff += psnr_tem
                ssim_tem = SSIM_function(gt.permute(2,0,1).unsqueeze(0), shaded_image_diff.permute(2,0,1).unsqueeze(0))
                ssim_diff += ssim_tem
                lpip_tem = loss_fn_vgg(gt.permute(2,0,1), shaded_image_diff.permute(2,0,1))
                lpip_vgg_diff_ = lpip_vgg_diff + lpip_tem
                print(f'Testing psnr(diff):{psnr_tem}--ssim:{ssim_tem}--lpip_vgg{lpip_tem} for view{str(view_count)}')

                

                masks,imgs_full,imgs_diff,imgs_spe,normals,sum1 = renderer.render(test_dataset,data, mesh_initial, channels=['mask', 'position', 'normal'], shader=shader,envmap=env_mapour) 

                shaded_image_envmapour = torch.clamp(imgs_full[0], 0, 1)  
                shaded_image_spe = torch.clamp(imgs_spe[0], 0, 1)   
                mseloss = MSE_function(shaded_image_envmapour.reshape(-1,3), gt.reshape(-1,3)).mean()
                psnr_tem = -10 * torch.log10(mseloss)
                psnr_envmapour += psnr_tem
                ssim_tem = SSIM_function(gt.permute(2,0,1).unsqueeze(0), shaded_image_envmapour.permute(2,0,1).unsqueeze(0))
                ssim_envmapour += ssim_tem
                lpip_tem = loss_fn_vgg(gt.permute(2,0,1), shaded_image_envmapour.permute(2,0,1))
                lpip_vgg_envmapour = lpip_tem + lpip_vgg_envmapour
                print(f'Testing psnr(envmapour):{psnr_tem}--ssim:{ssim_tem}--lpip_vgg{lpip_tem} for view{str(view_count)}')
                
    
                
                plt.imsave(shaded_path / f'view{str(view_count)}.png', shaded_image.cpu().numpy())
                plt.imsave(shaded_path_diff/ f'view{str(view_count)}.png', shaded_image_diff.cpu().numpy())
                plt.imsave(shaded_path_spe/ f'view{str(view_count)}.png', shaded_image_spe.cpu().numpy())
                plt.imsave(shaded_path_envmapour / f'view{str(view_count)}.png', shaded_image_envmapour.cpu().numpy())
                
                normal_path = (normal_img_path/'test')
                normal_path.mkdir(parents=True, exist_ok=True)
                normal_img = torch.clamp(normals[0],0,1)
                plt.imsave(normal_path / f'view{str(view_count)}.png', normal_img.cpu().numpy())
                
                imgs.append(shaded_image.cpu().numpy())
                dif.append(shaded_image_diff.cpu().numpy())
                spe.append(shaded_image_spe.cpu().numpy())
                imgs_envmapour.append(shaded_image_envmapour.cpu().numpy())

            
                view_count = view_count+1

        
            psnr = psnr/view_count
            ssim_full = ssim_full/view_count
            lpip_vgg = lpip_vgg/view_count
            print(f'Averaged testing psnr:{psnr}--ssim:{ssim_full}--lpip_vgg{lpip_vgg}') 
            
            psnr_diff = psnr_diff/view_count
            ssim_diff = ssim_diff/view_count
            lpip_vgg_diff = lpip_vgg_diff/view_count
            print(f'Averaged testing psnr(diffuse):{psnr_diff}--ssim:{ssim_diff}--lpip_vgg{lpip_vgg_diff}') 
            
            psnr_envmapour /=view_count
            ssim_envmapour /= view_count
            lpip_vgg_envmapour /=view_count
            print(f'Envmapour Averaged testing psnr:{psnr_envmapour}--ssim:{ssim_envmapour}--lpip_vgg{lpip_vgg_envmapour}')
            
                        
            imageio.mimwrite(experiment_dir/'FullShading.mp4', to8b(imgs), fps=30, quality=8)
            imageio.mimwrite(experiment_dir/'DiffuseShading.mp4', to8b(dif), fps=30, quality=8)
            imageio.mimwrite(experiment_dir/'SpecularShading.mp4', to8b(spe), fps=30, quality=8)
            imageio.mimwrite(experiment_dir/'EnvmapourFullshading.mp4', to8b(imgs_envmapour), fps=30, quality=8)
            
    if args.uvmap:
        utils.uvmapping(mesh_initial,args,shader)
   
    if args.test or args.uvmap:
        utils.write_mesh(meshes_path / f"mesh_test.obj", mesh_initial)
    if shader is not None and (args.test or args.uvmap):
        shader.save(shaders_save_path / f'shader_test.pt')
