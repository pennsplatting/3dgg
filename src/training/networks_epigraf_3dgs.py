import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from src.torch_utils import misc
from src.torch_utils import persistence
from src.dnnlib import EasyDict, TensorGroup
from omegaconf import DictConfig

from ipdb import set_trace as st 

from src.training.networks_stylegan2 import SynthesisBlock
from src.training.networks_camera_adaptor import CameraAdaptor
from src.training.networks_depth_adaptor import DepthAdaptor
from src.training.layers import (
    FullyConnectedLayer,
    MappingNetwork,
    ScalarEncoder1d,
)
from src.training.rendering_utils import compute_cam2world_matrix, compute_camera_intrinsics
from src.training.training_utils import linear_schedule, run_batchwise
from src.training.tri_plane_renderer import sample_rays, ImportanceRenderer, simple_tri_plane_renderer

import sys
sys.path.append('/home/zuoxy/cg/3dgp/submodules')
from gaussian_splatting.renderer import render as gs_render
from gaussian_splatting.cameras import MiniCam 
from gaussian_splatting.gaussian_model import GaussianModel

from src.debug import visualize_rays, project_rays_to_image
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisBlocksSequence(torch.nn.Module):
    # A simpler verion of the SG2 SynthesisNetwork, which can also take some 2d tensor as input.
    # This is useful to build a 2D upsampler.
    def __init__(self,
        cfg: DictConfig,            # Hyperparameters config.
        in_resolution,              # Which resolution do we start with?
        out_resolution,             # Output image resolution.
        in_channels,                # Number of input channels.
        out_channels,               # Number of input channels.
        num_fp16_res    = 4,        # Use FP16 for the N highest resolutions.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert in_resolution == 0 or (in_resolution >= 4 and math.log2(in_resolution).is_integer())
        assert out_resolution >= 4 and math.log2(out_resolution).is_integer()
        assert in_resolution < out_resolution
        super().__init__()
        self.cfg = cfg
        self.out_resolution = out_resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_fp16_res = num_fp16_res

        in_resolution_log2 = 2 if in_resolution == 0 else (int(np.log2(in_resolution)) + 1)
        out_resolution_log2 = int(np.log2(out_resolution))
        self.block_resolutions = [2 ** i for i in range(in_resolution_log2, out_resolution_log2 + 1)]
        out_channels_dict = {res: min(int(self.cfg.cbase * self.cfg.fmaps) // res, self.cfg.cmax) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (out_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for block_idx, res in enumerate(self.block_resolutions):
            cur_in_channels = out_channels_dict[res // 2] if block_idx > 0 else in_channels
            cur_out_channels = out_channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.out_resolution)
            block = SynthesisBlock(cur_in_channels, cur_out_channels, w_dim=self.cfg.w_dim, resolution=res,
                img_channels=self.out_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, x: torch.Tensor=None, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.cfg.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        cfg: DictConfig,            # Main config
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        depth_channels=1,
        **synthesis_seq_kwargs,     # Arguments of SynthesisBlocksSequence
    ):
        super().__init__()
        self.cfg = cfg
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.depth_channels = depth_channels

        decoder_out_channels = 96 #TODO: find suitable value for this

        self.tri_plane_decoder = SynthesisBlocksSequence(
            cfg=cfg,
            in_resolution=0,
            out_resolution=self.img_resolution,
            in_channels=0,
            out_channels=decoder_out_channels,
            architecture='skip',
            use_noise=self.cfg.use_noise,
            **synthesis_seq_kwargs,
        )

        self.options = EasyDict({'gen_xyz': True, 'gen_rgb': False, 'gen_sh': True, 'gen_opacity': True, 'gen_scaling': True, 'gen_rotation': True, 'gen_xyz_offset': False, 
        'xyz_offset_scale': 0.1, 'depth_bias': 2, 'depth_factor': 0.5, 
        'scale_bias': -2, 'scale_factor': 1, 'max_scaling': -1, 'min_scaling': -15})
        options = self.options
        self.feat_dim = 1 * options['gen_xyz'] + 3 * options['gen_rgb'] + 3 * options['gen_sh'] + 1 * options['gen_opacity'] + 3 * options['gen_scaling'] + 4 * options['gen_rotation'] + 3 * options['gen_xyz_offset']

        self.text_decoder = TextureDecoder(n_features=14, options=self.options)
        self.rgbd_head = nn.Conv2d(decoder_out_channels, img_channels + 1, kernel_size=3, stride=1, padding=0)
        # self.depth_head = nn.Conv2d(decoder_out_channels, depth_channels, kernel_size=3, stride=1, padding=0)

        # self.tri_plane_mlp = TriPlaneMLP(self.cfg, out_dim=self.img_channels)
        self.num_ws = self.tri_plane_decoder.num_ws
        # self.nerf_noise_std = 0.0
        self.train_resolution = self.cfg.patch.resolution if self.cfg.patch.enabled else self.img_resolution # TODO: enable patch resolution
        # self.train_resolution = self.img_resolution
        self.test_resolution = self.img_resolution
        self.z_near = 0.5
        self.z_far = 10 #TODO: fimd suitable value for this 
        h = w = (self.train_resolution if self.training else self.test_resolution)
        self.sh_degree = 0 #TODO: find suitable value for this
        self.viewpoint_camera = MiniCam(h, w, self.z_near, self.z_far) #TODO: resolution check 
        self.gaussian = GaussianModel(self.sh_degree)
        
        self.white_background = True
        bg_color = [1,1,1] if self.white_background else [0, 0, 0]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.background = torch.tensor(bg_color, dtype=torch.float32, device=self.device)

        self.ray_origins = None
        self.ray_directions = None
        # self.renderer = #TODO: Add renderer

        if self.cfg.depth_adaptor.enabled:
            self.depth_adaptor = DepthAdaptor(
                self.cfg.depth_adaptor,
                min_depth=self.cfg.camera.ray.start,
                max_depth=self.cfg.camera.ray.end
            )
        else:
            self.depth_adaptor = None

        if self.cfg.camera_adaptor.enabled:
            self.camera_adaptor = CameraAdaptor(self.cfg.camera_adaptor)
        else:
            self.camera_adaptor = None

        # Rendering options used at test time.
        # We overwrite them when we need to compute some additional losses
        self._default_render_options = EasyDict(
            max_batch_res=self.cfg.max_batch_res,
            return_depth=False,
            return_depth_adapted=False,
            return_weights=False,
            concat_depth=False,
            cut_quantile=0.0,
            density_bias=self.cfg.density_bias,
        )

    def progressive_update(self, cur_kimg: float):
        self.nerf_noise_std = linear_schedule(cur_kimg, self.cfg.nerf_noise_std_init, 0.0, self.cfg.nerf_noise_kimg_growth)
        if not self.depth_adaptor is None:
            self.depth_adaptor.progressive_update(cur_kimg)

    # @torch.no_grad()
    # def compute_densities(self, ws: torch.Tensor, coords: torch.Tensor, max_batch_res: int=32, **block_kwargs) -> torch.Tensor:
    #     plane_feats = self.tri_plane_decoder(ws[:, :self.tri_plane_decoder.num_ws], **block_kwargs) # [batch_size, 3 * feat_dim, tp_h, tp_w]
    #     ray_d_world = torch.zeros_like(coords) # [batch_size, num_points, 3]
    #     rgb_sigma = run_batchwise(
    #         fn=simple_tri_plane_renderer,
    #         data=dict(coords=coords),
    #         batch_size=max_batch_res ** 3,
    #         dim=1,
    #         mlp=self.tri_plane_mlp, x=plane_feats, scale=self.cfg.camera.cube_scale,p0) # [batch_size, num_coords, num_feats]

    #     return rgb_sigma['sigma'] # [batch_size, num_coords, 1]
        
    def update_gaussian(self, feature_gen): # use 1 splatter image at front view
        start_dim = 0

        if self.text_decoder.options['gen_xyz']:      
            _depth = feature_gen[:,start_dim:start_dim+1]

            self.gaussian.update_xyz(_depth, self.ray_origins, self.ray_directions)
            start_dim += 1
        
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

    def forward(self, ws, camera_params: EasyDict[str, torch.Tensor], patch_params: Dict=None, render_opts: Dict={}, **block_kwargs):
        """
        ws: [batch_size, num_ws, w_dim] --- latent codes
        camera_params: EasyDict {angles: [batch_size, 3], fov: [batch_size], radius: [batch_size], look_at: [batch_size, 3]} --- camera parameters
        patch_params: Dict {scales: [batch_size, 2], offsets: [batch_size, 2]} --- patch parameters (when we do patchwise training)
        """
        render_opts = EasyDict(**{**self._default_render_options, **render_opts})
        batch_size, num_steps = ws.shape[0], self.cfg.num_ray_steps
        decoder_out = self.tri_plane_decoder(ws[:, :self.tri_plane_decoder.num_ws], **block_kwargs) # [batch_size, 96, 512, 512]
        # print("Decoder out shape:", decoder_out.shape)

        # decoder_out = self.gs_decoder() #TODO: Add decoder: (batch_size, num_ws, w_dim) => (batch_size, )
        # h, w = self.img_resolution, self.img_resolution
        h = w = (self.train_resolution if self.training else self.test_resolution)
        # print("Training enabled:", self.training)
        # print("Train resolution:", self.train_resolution)
        # print("H:", h, "W:", w)
        # plane_feats = decoder_out[:, :3 * self.cfg.tri_plane.feat_dim].view(batch_size, 3, self.cfg.tri_plane.feat_dim, h, w)
        # print(plane_feats.shape)
        feature_images = self.text_decoder(decoder_out) # [batch_size, feat_dim, img_resolution, img_resolution]
        if self.training:
            feature_images = F.interpolate(feature_images, size=(h, w), mode='bilinear', align_corners=False) # [batch_size, feat_dim, h, w]
        # feature_images = self.text_decoder(decoder_out) # [batch_size, feat_dim, h, w]
        # print("Feature images shape:", feature_images.shape)

        imgs = []
        depths = []
        alphas = []

        c2ws = compute_cam2world_matrix(camera_params) # [batch_size, 4, 4]
        intrinsics = compute_camera_intrinsics(camera_params, w, h) # [batch_size, 3, 3]
        ray_origins, ray_directions = sample_rays(c2ws, fov=camera_params.fov, resolution=(h, w), patch_params=patch_params, device=ws.device) # [batch_size, h, w, 3]
        # print("focal length:", intrinsics[:, 0, 0])
        # print(intrinsics[0, 0, 1])
        # print("C2Ws shape:", c2ws.shape)
        # print("Ray origins shape:", ray_origins.shape)
        # # print("H:", h, "W:", w)
        # for c2w, intrinsic, feature_image, ray_o, ray_d in zip(c2ws, intrinsics, feature_images, ray_origins, ray_directions):
        #     feature_image = feature_image.reshape(-1, self.feat_dim)
        #     # print("Feature image min:", feature_image.min(), "max:", feature_image.max(), "mean:", feature_image.mean())
        #     # print("Feature image shape:", feature_image.shape)
        #     self.ray_origins, self.ray_directions = ray_o, ray_d 
        #     self.ray_directions = self.ray_directions / self.ray_directions.norm(dim=-1, keepdim=True)
        #     print("Ray origin shape:", self.ray_origins.shape)
        #     # print("Ray directions shape:", self.ray_directions.shape)
        #     print("ray origins min:", self.ray_origins.min(), "max:", self.ray_origins.max())
        #     print("ray directions min:", self.ray_directions.min(), "max:", self.ray_directions.max())
        #     print(f"Z-direction min: {self.ray_directions[:, 2].min()}, max: {self.ray_directions[:, 2].max()}")
        #     # visualize_rays(self.ray_origins, self.ray_directions, scale=1.0)
        #     # project_rays_to_image(self.ray_origins, self.ray_directions, img_size=w, z_plane=2.0)
        #     # project_rays_to_image(self.ray_origins, self.ray_directions, img_size=w, z_plane=3.0)
        #     # project_rays_to_image(self.ray_origins, self.ray_directions, img_size=w, z_plane=1.0)
        #     # break
        #     self.viewpoint_camera.update_transforms2(intrinsic, c2w)
        #     self.update_gaussian(feature_image)
        #     res = gs_render(self.viewpoint_camera, self.gaussian, None, self.background)
        #     img = res['render']
        #     depth = res['depth']
        #     alpha = res['alpha']
        #     imgs.append(img[None])
        #     depths.append(depth[None])
        #     alphas.append(alpha[None])
        # img = torch.cat(imgs, dim=0)
        # depth = torch.cat(depths, dim=0)
        rgbd = self.rgbd_head(decoder_out)
        img = rgbd[:, :self.img_channels]  
        depth = rgbd[:, self.img_channels:]
        img = F.interpolate(img, size=(h, w), mode='bilinear', align_corners=False)
        depth = F.interpolate(depth, size=(h, w), mode='bilinear', align_corners=False)
        # print("Depth adapted enabld:", render_opts.return_depth_adapted)
        # print("Depth enabled:", render_opts.return_depth)
        # print("Depth adaptor:", self.depth_adaptor)
        # print("Concat depth:", render_opts.concat_depth)
        if not self.depth_adaptor is None:
            # depth_adapted = self.depth_adaptor(depth, ws[:, 0]) # [batch_size, 1, h, w]
            # print("Minimum adapted depth:", depth_adapted.min().item(), "Maximum adapted depth:", depth_adapted.max().item())

            if render_opts.concat_depth:
                # img = torch.cat([img, depth_adapted], dim=1) # [batch_size, c + 1, h, w]
                img = torch.cat([img, depth], dim=1) # [batch_size, c + 1, h, w]
            else:
                # To avoid potential DataParallel issues
                # img = img + 0.0 * depth_adapted.max() # [batch_size, c, h, w]
                img = img + 0.0 * depth.max()
        # print("Img shape after concat:", img.shape)
        if render_opts.return_depth or render_opts.return_depth_adapted:
            out = TensorGroup(img=img)
            if render_opts.return_depth: out.depth = depth # [batch_size, 1, h, w]  
            if render_opts.return_depth_adapted: out.depth_adapted = depth
            # print("Output img shape:", out.img.shape)
            # print("Output depth shape:", out.depth.shape)
            return out
        else:
            return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        cfg: DictConfig,            # Main config
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.cfg = cfg
        self.z_dim = self.cfg.z_dim
        self.c_dim = self.cfg.c_dim
        self.w_dim = self.cfg.w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.depth_channels = 1
        self.synthesis = SynthesisNetwork(cfg=cfg, img_resolution=img_resolution, img_channels=img_channels, depth_channels=self.depth_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=self.cfg.z_dim, c_dim=self.cfg.c_dim, w_dim=self.cfg.w_dim, num_ws=self.num_ws, camera_raw_scalars=True, num_layers=self.cfg.map_depth, **mapping_kwargs)

    def progressive_update(self, cur_kimg: float):
        self.synthesis.progressive_update(cur_kimg)

    def forward(self, z, c, camera_params, camera_angles_cond=None, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, camera_angles=camera_angles_cond, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        out = self.synthesis(ws, camera_params=camera_params, update_emas=update_emas, **synthesis_kwargs)
        return out

#----------------------------------------------------------------------------

class TextureDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.options = options
        self.out_dim = 1 * options['gen_xyz'] + 3 * options['gen_rgb'] + 3 * options['gen_sh'] + 1 * options['gen_opacity'] + 3 * options['gen_scaling'] + 4 * options['gen_rotation'] + 3 * options['gen_xyz_offset']
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
        # features (4, 96, 256, 256) -> (4, 256, 256, 96)
        # Aggregate features
        sampled_features = sampled_features.permute(0,2,3,1)
        x = sampled_features

        N, H, W, C = x.shape
        # x = x.reshape(N*H*W, C)
        
        # # x = self.net(x) # TODO: delecte the net 
        # x = x.reshape(N, H, W, -1)

        start_dim = 0
        out = {}
        if self.options['gen_xyz']:
            out['depth'] = self.depth_bias + self.depth_factor * torch.nn.functional.normalize(x[..., start_dim:start_dim+1])
            start_dim += 1

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