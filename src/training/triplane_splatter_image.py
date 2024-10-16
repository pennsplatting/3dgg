# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
from torch_utils import persistence
from training.networks_stylegan2 import Generator as StyleGAN2Backbone
from training.volumetric_rendering.renderer import ImportanceRenderer
from training.volumetric_rendering.ray_sampler import RaySampler
import dnnlib

from ipdb import set_trace as st
import cv2
import torch.nn as nn
import torch.nn.functional as F

from training.gaussian_splatting.gaussian_model import GaussianModel
from training.gaussian_splatting.cameras import MiniCam
from training.gaussian_splatting.renderer import render as gs_render
from training.gaussian_splatting.utils.graphics_utils import getWorld2View, getProjectionMatrix
import numpy as np

from plyfile import PlyData
import trimesh
import scipy.io as sio
from torch_utils.extract_edge import EdgeExtractor
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from training.bg_utils import hemispherical_texture_map
import matplotlib.pyplot as plt


def print_grad(name, grad):
    print(f"{name}:")
    if torch.all(grad==0):
        print("grad all 0s")
        return 
    # print(grad)
    print('\t',grad.max(), grad.min(), grad.mean())


@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        plane_resolution,
        edge_discriminate   = True,
        use_template        = True,
        multi_splatter      = False,
        sphere_bg           = False,
        sh_degree           = 0,    # Spherical harmonics degree.
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        text_decoder_kwargs = {},   # GS TextureDecoder
        sr_kwargs = {},
        device              = 'cuda',  
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.renderer = ImportanceRenderer()
        self.ray_sampler = RaySampler()
        # TODO: maybe synthesize uv texture directly with stylegan2 backbone (img_channels=3)
        # so that there's no need for decoder
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=plane_resolution, img_channels=14, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self.sphere_bg = sphere_bg
        if sphere_bg:
            self.backbone_bg = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=img_resolution, img_channels=3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        # self.text_decoder = TextureDecoder(96, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': (sh_degree + 1) ** 2 * 3})
        text_decoder_options = {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 0}
        text_decoder_options.update(text_decoder_kwargs)

        # self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=self.text_decoder.out_dim, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        
        self.neural_rendering_resolution = 64
        self.plane_resolution = plane_resolution
        self.rendering_kwargs = rendering_kwargs
        
        self._last_planes = None
                
        self.gaussian_splatting_use_sr = False
        self.sh_degree = sh_degree
        self.edge_discriminate = edge_discriminate
        self.use_template = use_template
        self.multi_splatter = multi_splatter
        self.device = device
        # self.gaussian = None
        # self.viewpoint_camera = None
        
        # initialize camera and gaussians
        
        ## gaussian_splatting rendering resolution: 
        ## V1: gaussian rendering -> 64 x 64 -> sr module --> 256 x 256: self.gaussian_splatting_use_sr = True
        ## v2: gausian rendering -> 256 x 256 : self.gaussian_splatting_use_sr = False
        if self.gaussian_splatting_use_sr:
            image_size = self.neural_rendering_resolution # chekc
        else:
            image_size = img_resolution
        self.z_near = 0.1
        self.z_far = 4 # TODO: find suitable value for this
        self.viewpoint_camera = MiniCam(image_size, image_size, self.z_near, self.z_far)
        
        ### -------- gaussian splatting render --------
        self.gaussian = GaussianModel(self.sh_degree)
        # self.depth_image = None
        self.viewpoint_camera2 = MiniCam(plane_resolution, plane_resolution, self.z_near, self.z_far)
        
        # if self.multi_splatter:
        #     cam_pivot = torch.tensor(rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device="cuda")
        #     cam_radius = rendering_kwargs.get('avg_camera_radius', 2.7)
        #     angle_y = -0.4
        #     angle_p = -0.2
        #     fov_deg = 18.837
        #     c2w_gen_left = LookAtPoseSampler.sample(np.pi/2 - angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device="cuda")
        #     c2w_gen_right = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device="cuda")
        #     intrinsics_gen = FOV_to_intrinsics(fov_deg, device="cuda")
        #     ray_origins, ray_directions = self.ray_sampler(c2w_gen_left, intrinsics_gen[None], self.plane_resolution)
        #     self.ray_origins = ray_origins
        #     self.ray_directions = ray_directions
        #     ray_origins, ray_directions = self.ray_sampler(c2w_gen_right, intrinsics_gen[None], self.plane_resolution)
        #     self.ray_origins = torch.cat((self.ray_origins, ray_origins), dim=1).squeeze()
        #     self.ray_directions = torch.cat((self.ray_directions, ray_directions), dim=1).squeeze()
        #     # self.sphere_mapping()
        #     self.viewpoint_camera2.update_transforms2(intrinsics_gen, c2w_gen_left[0])
        #     self.depth_image = self.depth_of_object(self.viewpoint_camera2)
        #     self.viewpoint_camera2.update_transforms2(intrinsics_gen, c2w_gen_right[0])
        #     self.depth_image = torch.cat((self.depth_image, self.depth_of_object(self.viewpoint_camera2)), dim=0)
        #     self.text_decoder = TextureDecoder2View(96, text_decoder_options)
        else:
            c2w_gen = torch.eye(4)[None].to(device) 
            c2w_gen[:,1:3] *= -1 # lookat: neg z
            c2w_gen[:,2,3] = 2.69 # z position
            intrinsics_gen = torch.tensor([[[2.5000, 0.0000, 0.5000],
                                            [0.0000, 2.5000, 0.5000],
                                            [0.0000, 0.0000, 1.0000]]]).to(device)
            # ray_origins, ray_directions = self.ray_sampler(c2w_gen, intrinsics_gen, self.plane_resolution)
            # self.ray_origins = ray_origins[0].to(device)
            # self.ray_directions = ray_directions[0].to(device)
            
            # self.register_buffer('ray_origins', ray_origins[0].contiguous())
            # self.register_buffer('ray_directions', ray_directions[0].contiguous())
            

            # self.viewpoint_camera2.update_transforms2(intrinsics_gen[0], c2w_gen[0])
            # if self.use_template:
            #     self.depth_image = self.depth_of_object(self.viewpoint_camera2)
            self.text_decoder = TextureDecoder(14, text_decoder_options)

        if self.edge_discriminate:
            self.edge_extractor = EdgeExtractor().cuda()

        white_background = True
        bg_color = [1,1,1] if white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device=device)
        print("Generator init completed.")
        
    def sphere_mapping(self):
        print(self.ray_directions.shape, f"self.ray_directions: {self.ray_directions}")
        exit(0)
        
    def generate_coordinates(self, img_resolution):
        x = torch.arange(0, img_resolution, dtype=torch.float32)
        y = torch.arange(0, img_resolution, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        coordinates = torch.stack([xx, yy], dim=0) / img_resolution # [2, res, res]
        
        zeros = torch.zeros((1, img_resolution, img_resolution), dtype=torch.float32)
        result = torch.cat([coordinates, zeros], dim=0)
        return result

    def filter_uv(self, uv_coords, vertices, threshold):
        boundary_uvs = ((uv_coords > threshold) | (uv_coords < - threshold)).any(axis=1).squeeze()
        filtered_uvs = uv_coords[~boundary_uvs]
        filtered_verts = vertices[~boundary_uvs]
        return filtered_uvs, filtered_verts

    def process_uv(self, uv_coords, uv_h = 256, uv_w = 256):
        return uv_coords*2-1.0
    
    def depth_of_object(self, camera : MiniCam):
        # Transform object coordinates to camera coordinates
        object_coords_homogeneous = torch.cat((self.verts, torch.ones(self.verts.shape[0], 1)), dim=1).to(self.device)
        # print(object_coords_homogeneous.device, camera.full_proj_transform.device)
        # # Apply perspective projection
        object_coords_camera = torch.matmul(object_coords_homogeneous, camera.full_proj_transform)
        
        # Apply perspective division
        projected_coords = object_coords_camera[:, :2] / object_coords_camera[:, 2:]

        # Normalize points to image coordinates
        projected_coords = (projected_coords + 1) * torch.tensor([camera.image_width, camera.image_height], device=self.device)[None, :] / 2
        
        # Initialize depth image with maximum depth value
        depth_image = torch.full((camera.image_height, camera.image_width), self.z_far, dtype=torch.float32, device=self.device)
        # Calculate depth for each projected point and update depth image
        for i in range(self.verts.shape[0]):
            u, v = projected_coords[i, :2].round().int()
            if 0 <= u < camera.image_width and 0 <= v < camera.image_height:
                # depth = (2 * self.z_far * self.z_near) / (self.z_far + self.z_near - projected_coords[i, 2] * (self.z_far - self.z_near))
                depth_image[v, u] = min(object_coords_camera[i, 2], depth_image[v, u])
        return depth_image.reshape(-1, 1)
        
    def update_gaussian(self, feature_gen): # use 1 splatter image at front view
        start_dim = 0

        # if self.use_template:
        #     _depth = torch.where(
        #         self.depth_image < 3.0,
        #         self.depth_image, # if true 
        #         feature_gen[:,start_dim:start_dim+1]) # if false
        # else:
        #     _depth = feature_gen[:,start_dim:start_dim+1]

        # self.gaussian.update_xyz(_depth, self.ray_origins, self.ray_directions)
       
        # start_dim += 1
        if self.text_decoder.options['gen_xyz_offset']:
            _xyz_offset = feature_gen[:,start_dim:start_dim+3]
            self.gaussian.update_xyz_offset(self.verts, _xyz_offset)
            # print(f"xyz_offset: min={_xyz_offset.min(dim=0)}, max={_xyz_offset.max(dim=0)}")
            # print(f"verts: min={self.verts.min(dim=0)}, max={self.verts.max(dim=0)}")
            # st()
            start_dim += 3

        if self.text_decoder.options['gen_rgb']:
            # print(feature_gen[:,start_dim].shape)
            _rgb = feature_gen[:,start_dim:start_dim+3]
            start_dim += 3
            self.gaussian.update_rgb_textures(_rgb)
        
        if self.text_decoder.options['gen_sh']:
            _sh = feature_gen[:,start_dim:start_dim+3]
            start_dim += 3
            self.gaussian.update_sh_texture(_sh)
        
        if self.text_decoder.options['gen_opacity']:
            _opacity = feature_gen[:,start_dim:start_dim+1] # should be no adjustment for sigmoid
            start_dim += 1
            self.gaussian.update_opacity(_opacity)
        
        if self.text_decoder.options['gen_scaling']:
            _scaling = feature_gen[:,start_dim:start_dim+3]
            self.gaussian.update_scaling(_scaling, max_s = self.text_decoder.options['max_scaling'], min_s = self.text_decoder.options['min_scaling'])
            start_dim += 3
            
        if self.text_decoder.options['gen_rotation']:
            _rotation = feature_gen[:,start_dim:start_dim+4]
            self.gaussian.update_rotation(_rotation)
            start_dim += 4
        

    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        ws = self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return ws
            
    def synthesis(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)
        
        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution
    
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
            if self.sphere_bg:
                bg_plane = self.backbone_bg.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
                bg_plane = torch.sigmoid(bg_plane)*(1 + 2*0.001) - 0.001 # [0, 1]
        if cache_backbone:
            self._last_planes = planes

    
        ## FIXME: replace with gt texture for debug 
        # textures = torch.tensor(get_uv_texture(), dtype=torch.float, device="cuda") # (53215, 3)
        
        # original_eg3d = False
        # if original_eg3d:
        #     ## ----- original eg3d version -----
        #     # Create a batch of rays for volume rendering
        #     ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        #     # Create triplanes by running StyleGAN backbone
        #     N, M, _ = ray_origins.shape
            
        #     planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1]) # torch.Size([4, 3, 32, 256, 256])

        #     # Perform volume rendering
        #     feature_samples, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last
        #     # feature_samples: [4, 4096, 32]

        #     # Reshape into 'raw' neural-rendered image
        #     H = W = self.neural_rendering_resolution
        #     feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        #     depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        #     # Run superresolution to get final image
        #     rgb_image = feature_image[:, :3] # rgb: torch.Size([4, 3, 64, 64]), feature_image: torch.Size([4, 32, 64, 64])
        #     st() # check feature image.shape
        #     sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
            
        #     ### ----- original eg3d version [END] ----- 
        
        
        else:
            
            ### ----- gaussian splatting -----
          
            feature_image = self.text_decoder(planes)
            B, K, H, W = feature_image.shape
            # if self.multi_splatter:
            #     feature_gen_batch = feature_image
            # else: 
            #     B, K, H, W = feature_image.shape
            #     feature_gen_batch = feature_image.permute(0,2,3,1).reshape(B, H*W, K).contiguous()
    
            grid = self.uvcoords.unsqueeze(-2).repeat(B,1,1,1) # [B,5023,1,2]
            #debug_texture
            # feature_image[:, 3:6] = torch.zeros_like(feature_image[:, 3:6])+0.9
            # x_, y_ = torch.arange(W), torch.arange(H)
            # X_, Y_ = torch.meshgrid(x_, y_)
            # feature_image[:, 3] = X_ / W
            # feature_image[:, 4] = Y_ / H
            # color_grid = self.uvcoords.unsqueeze(-2)
            # color_map = feature_image[0, 3:6].unsqueeze(0)
            # print(color_map.shape, color_grid.shape)
            # colors = F.grid_sample(color_map, color_grid, align_corners=False).squeeze().permute(1, 0).detach().cpu().numpy()
            # uv = np.clip((self.uvcoords.squeeze().cpu().numpy()+1)*256, 0, 511)
            # print(colors.shape, colors.max(), colors.min())
            # print(uv.shape, uv.max(), uv.min())
            # st()
            # plt.imshow(np.clip(feature_image[0, 3:6].detach().cpu().numpy().transpose(1,2,0), 0, 1))
            # plt.savefig("/home/zuoxy/eg3d/eg3d/test/texture_debug.png")
            # plt.scatter(uv[:,0], uv[:,1], c=colors, s=2)
            # plt.savefig("/home/zuoxy/eg3d/eg3d/test/uv_debug.png")

            texture_batch = F.grid_sample(feature_image, grid, align_corners=False).squeeze().permute(0,2,1) # [B, 14, 5023, 1] -> [B, 5023, 14]

            rgb_image_batch = []
            alpha_image_batch = [] # mask
            depth_image_batch = []

            uv_image_batch = []
            opacity_activated_batch = []
            offset_batch = []
            scale_activated_batch = []
            

            # c2w_gen = torch.eye(4)[None].repeat(feature_gen_batch.shape[0],1,1).to(feature_gen_batch.device)
            # c2w_gen[:,1:3] *= -1 # lookat: neg z
            # c2w_gen[:,2,3] = 2.69 # z position
            # intrinsics_gen = torch.tensor([[[3.0000, 0.0000, 0.5000],
            #                                 [0.0000, 3.0000, 0.5000],
            #                                 [0.0000, 0.0000, 1.0000]]]).repeat(feature_gen_batch.shape[0],1,1).to(feature_gen_batch.device)

            # ray_origins, ray_directions = self.ray_sampler(c2w_gen, intrinsics_gen, self.plane_resolution)
            
            # self.viewpoint_camera2.update_transforms2(intrinsics_gen[0], c2w_gen[0])
            # with torch.no_grad(): # NOTE: do not update head template
            #     # FIXME: this may not be the best way to get depth, because the edge is not clean 
            #     # depth_image = gs_render(self.viewpoint_camera2, self.face_model, None, self.background)["depth"] # [1, H, W]
            #     # depth_image = depth_image.permute(1,2,0).reshape(-1, 1).contiguous()
            #     depth_image = self.depth_of_object()
            #     depth_image = depth_image.reshape(-1, 1)
            
            # print(f"--textures_gen_batch: min={textures_gen_batch.min()}, max={textures_gen_batch.max()}, mean={textures_gen_batch.mean()}, shape={textures_gen_batch.shape}")
            
            # if self.sphere_bg:
            #     bg_batch = []
            #     for _cam2world_matrix, _intrinsics, feature_gen, bg_gen, ray_direction in zip(cam2world_matrix, intrinsics, feature_gen_batch, bg_plane, ray_directions):
            #         self.viewpoint_camera.update_transforms2(_intrinsics, _cam2world_matrix)

            #         ## TODO: can gaussiam splatting run batch in parallel?
            #         # textures = F.grid_sample(texture_gen[None], self.raw_uvcoords.unsqueeze(1), align_corners=False) # (1, 48, 1, 5023)
            #         # textures.requires_grad_(True) 
            #         # textures.register_hook(lambda grad: print_grad("--textures.requires_grad", grad))

            #         self.update_gaussian(feature_gen)
                                
            #         res = gs_render(self.viewpoint_camera, self.gaussian, None, self.background)
            #         _rgb_image = res["render"]
            #         _alpha_image = res["alpha"]
            #         _depth_image = res["depth"]
                    
            #         _bg = hemispherical_texture_map(ray_direction, bg_gen)
            #         # print(_bg.max(), _bg.min())
            #         _rgb_image = _alpha_image * _rgb_image + (1-_alpha_image) * _bg
                    
            #         bg_batch.append(_bg[None])
            #         rgb_image_batch.append(_rgb_image[None])
            #         alpha_image_batch.append(_alpha_image[None])
            #         depth_image_batch.append(_depth_image[None])

            #         # Opacity for beta reg
            #         opacity_activated = self.gaussian.get_opacity
            #         opacity_activated_batch.append(opacity_activated)

            else:
                # for _cam2world_matrix, _intrinsics, feature_gen in zip(cam2world_matrix, intrinsics, feature_gen_batch):
                for _cam2world_matrix, _intrinsics, texture in zip(cam2world_matrix, intrinsics, texture_batch):
                    self.viewpoint_camera.update_transforms2(_intrinsics, _cam2world_matrix)

                    ## TODO: can gaussiam splatting run batch in parallel?
                    # textures = F.grid_sample(texture_gen[None], self.raw_uvcoords.unsqueeze(1), align_corners=False) # (1, 48, 1, 5023)
                    # textures.requires_grad_(True) 
                    # textures.register_hook(lambda grad: print_grad("--textures.requires_grad", grad))

                    self.update_gaussian(texture) # [5032, 14]
                    res = gs_render(self.viewpoint_camera, self.gaussian, None, self.background)
                    _rgb_image = res["render"]
                    _alpha_image = res["alpha"]
                    _depth_image = res["depth"]
                    
                    rgb_image_batch.append(_rgb_image[None])
                    alpha_image_batch.append(_alpha_image[None])
                    depth_image_batch.append(_depth_image[None])
                    
                    # Opacity for beta reg
                    opacity_activated = self.gaussian.get_opacity
                    opacity_activated_batch.append(opacity_activated)

                    scale_activated = self.gaussian.get_scaling
                    scale_activated_batch.append(scale_activated)
                
                    offset = self.gaussian.offset
                    offset_batch.append(offset)

            
            rgb_image = torch.cat(rgb_image_batch) # [4, 3, gs_res, gs_res]
            alpha_image = torch.cat(alpha_image_batch)
            depth_image = torch.cat(depth_image_batch)
            ## FIXME: try different normalization method to normalize rgb image to [-1,1]
            rgb_image = (rgb_image - 0.5) * 2
            alpha_image = (alpha_image - 0.5) * 2

            # uv_image = feature_image[:,3:6,:,:] # 0~1
            # print("max ", uv_image.max(), " min ", uv_image.min())
            # uv_image = torch.cat(uv_image_batch) 
            # uv_image = (uv_image - 0.5) * 2

            opacities = torch.cat(opacity_activated_batch)
            scale = torch.cat(scale_activated_batch)
            offset = torch.cat(offset_batch)

            # print(f"-rgb_image: min={rgb_image.min()}, max={rgb_image.max()}, mean={rgb_image.mean()}, shape={rgb_image.shape}")
            
            # rgb_image.requires_grad_(True)
            # rgb_image.register_hook(lambda grad: print_grad("rgb_image.requires_grad", grad))
            
            ## TODO: the below superresolution shall be kept?
            ## currently keeping the sr module below. TODO: shall we replace the feature image by texture_uv_map or only the sampled parts?
            if self.gaussian_splatting_use_sr:
                sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
            else:
                sr_image = rgb_image
                rgb_image = rgb_image[:,:,::8, ::8] # TODO: FIXME change this downsample to a smoother gaussian filtering
            
            if self.edge_discriminate:
                image_edge = self.edge_extractor(sr_image)
            ## TODO: render depth_image. May not explicitly calculated from the face model since its the same for all.
            # depth_image = torch.zeros_like(rgb_image) # (N, 1, H, W)

            if self.sphere_bg:
                bg = torch.cat(bg_batch)
                bg = (bg - 0.5) * 2
            ### ----- gaussian splatting [END] -----
        if self.sphere_bg:
            return {'image': sr_image, 'image_raw': rgb_image, 'image_mask': alpha_image, 'image_depth': depth_image, 'image_edge': image_edge, 'image_bg': bg, 'uv_image': uv_image, 'opacities': opacities}
        else:
            return {'image': sr_image, 'image_raw': rgb_image, 'image_mask': alpha_image, 'image_depth': depth_image, 'uv_image': uv_image, 'opacities': opacities, 'M_pos': offset, 'scale': scale}
    
    def sample(self, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        # return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)
        ## TODO: map the planes to the 3DMM model
        st()
        print('')
        face_model = tdmmModel.fit_feature_planes(planes)
        ## TODO2: check whether the self.decoder can be kept or not
        return self.renderer.run_model(face_model, self.decoder, coordinates, directions, self.rendering_kwargs)


    def sample_mixed(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws, update_emas = update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)


from training.networks_stylegan2 import FullyConnectedLayer

class TextureDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.options = options
        self.out_dim = 1 + 3 * options['gen_rgb'] + 3 * options['gen_sh'] + 1 * options['gen_opacity'] + 3 * options['gen_scaling'] + 4 * options['gen_rotation'] + 3 * options['gen_xyz_offset']
        self.xyz_offset_scale = options['xyz_offset_scale']
        self.depth_bias = options['depth_bias']
        self.depth_factor = options['depth_factor']
        self.scale_bias = options['scale_bias']
        self.scale_factor = options['scale_factor']

        # self.net = torch.nn.Sequential(
        #     FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
        #     torch.nn.Softplus(),
        #     FullyConnectedLayer(self.hidden_dim, self.out_dim, lr_multiplier=options['decoder_lr_mul'])
        # )
        
        self.offset_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.pos_act = nn.Tanh() #TODO: find best offset threshold for pos_act 
        
        # init weights as zeros
        nn.init.constant_(self.offset_conv.weight, 0)
        if self.offset_conv.bias is not None:
            nn.init.constant_(self.offset_conv.bias, 0)
        
    def forward(self, sampled_features):
        # features (4, 96, 256, 256) -> (4, 16*3, 256, 256)
        # Aggregate features
        sampled_features = sampled_features.permute(0,2,3,1)
        x = sampled_features

        N, H, W, C = x.shape
        # x = x.reshape(N*H*W, C)
        
        # # x = self.net(x) # TODO: delecte the net 
        # x = x.reshape(N, H, W, -1)

        start_dim = 0
        out = {}

        # out['depth'] = self.depth_bias + self.depth_factor * torch.nn.functional.normalize(x[..., start_dim:start_dim+1])
        # start_dim += 1
        if self.options['gen_xyz_offset']:
            # out['xyz_offset'] = self.xyz_offset_scale * torch.nn.functional.normalize(x[..., start_dim:start_dim+3]) # TODO: whether use this normalize? May constrain the offset not deviate too much
            out['xyz_offset'] = self.pos_act(self.offset_conv(x[..., start_dim:start_dim+3].permute(0,3,1,2)).permute(0,2,3,1))
            start_dim += 3

        if self.options['gen_rgb']:
            out['rgb'] = torch.sigmoid(x[..., start_dim:start_dim+3])*(1 + 2*0.001) - 0.001
            # print("max ", out['rgb'].max(), " min ", out['rgb'].min())
            start_dim += 3
        
        if self.options['gen_sh']:
            out['sh'] = x[..., start_dim:start_dim+3]
            start_dim += 3
        
        if self.options['gen_opacity']:
            out['opacity'] = x[..., start_dim:start_dim+1] # should be no adjustment for sigmoid
            start_dim += 1
        
        if self.options['gen_scaling']:
            # out['scaling'] = self.scale_bias + x[..., start_dim:start_dim+3].reshape(N, H, W, 3)
            out['scaling'] = self.scale_bias + self.scale_factor * torch.nn.functional.normalize(x[..., start_dim:start_dim+3]).reshape(N, H, W, 3)
            # out['scaling'] = torch.clamp(torch.exp(x[..., start_dim:start_dim+3].reshape(-1,3)), max=self.options['max_scaling']).reshape(N, H, W, 3)
            start_dim += 3
            
        if self.options['gen_rotation']:
            out['rotation'] = torch.nn.functional.normalize(x[..., start_dim:start_dim+4].reshape(-1,4).reshape(N, H, W, 4)) # check consistency before/after normalize: passed. Use: x[2,2,3,7:11]/out['rotation'][2,:,2,3]
            start_dim += 4


        # x.permute(0, 3, 1, 2)
        for key, v in out.items():
            # print(f"{key}:{v.shape}")
            out[key] = v.permute(0, 3, 1, 2)
            # print(f"{key} reshape to -> :{out[key].shape}")

        out_planes = torch.cat([v for key, v in out.items()] ,dim=1)
        return out_planes
    
class TextureDecoder2View(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.options = options
        self.out_dim = (1 + 3 * options['gen_rgb'] + 3 * options['gen_sh'] + 1 * options['gen_opacity'] + 3 * options['gen_scaling'] + 4 * options['gen_rotation'] + 3 * options['gen_xyz_offset']) * 2
        self.xyz_offset_scale = options['xyz_offset_scale']
        self.depth_bias = options['depth_bias']
        self.depth_factor = options['depth_factor']
        self.scale_bias = options['scale_bias']
        self.scale_factor = options['scale_factor']

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, self.out_dim, lr_multiplier=options['decoder_lr_mul'])
        )
        
    def forward(self, sampled_features):
        # Aggregate features
        sampled_features = sampled_features.permute(0,2,3,1)
        x = sampled_features

        N, H, W, C = x.shape
        x = x.reshape(N*H*W, C)
        
        x = self.net(x)
        x = x.reshape(N, H*W, -1)
        x_left = x[...,:self.out_dim//2]
        x_right = x[...,self.out_dim//2:]

        x = torch.cat((x_left, x_right), dim=1) # [4, 128, 128, 30]


        start_dim = 0
        out = {}

        out['depth'] = self.depth_bias + self.depth_factor * torch.nn.functional.normalize(x[..., start_dim:start_dim+1])
        start_dim += 1

        if self.options['gen_rgb']:
            out['rgb'] = torch.sigmoid(x[..., start_dim:start_dim+3])*(1 + 2*0.001) - 0.001
            start_dim += 3
        
        if self.options['gen_sh']:
            out['sh'] = x[..., start_dim:start_dim+3]
            start_dim += 3
        
        if self.options['gen_opacity']:
            out['opacity'] = x[..., start_dim:start_dim+1] # should be no adjustment for sigmoid
            start_dim += 1
        
        if self.options['gen_scaling']:
            out['scaling'] = self.scale_bias + self.scale_factor * torch.nn.functional.normalize(x[..., start_dim:start_dim+3])
            # out['scaling'] = torch.clamp(torch.exp(x[..., start_dim:start_dim+3].reshape(-1,3)), max=self.options['max_scaling']).reshape(N, H, W, 3)
            start_dim += 3
            
        if self.options['gen_rotation']:
            out['rotation'] = torch.nn.functional.normalize(x[..., start_dim:start_dim+4]) # check consistency before/after normalize: passed. Use: x[2,2,3,7:11]/out['rotation'][2,:,2,3]
            start_dim += 4
        
        if self.options['gen_xyz_offset']:
            out['xyz_offset'] = self.xyz_offset_scale * torch.nn.functional.normalize(x[..., start_dim:start_dim+3]) # TODO: whether use this normalize? May constrain the offset not deviate too much
            start_dim += 3


        # x.permute(0, 3, 1, 2)
        # for key, v in out.items():
        #     # print(f"{key}:{v.shape}")
        #     out[key] = v.permute(0, 3, 1, 2)
        #     # print(f"{key} reshape to -> :{out[key].shape}")

        out_planes = torch.cat([v for key, v in out.items()] ,dim=2)
        return out_planes
