import zipfile
from PIL import Image 
from io import BytesIO
import os
import torch
import cv2
import numpy as np
import sys
from PIL import Image
from submodules.depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2
# from submodules.unidepth.unidepth.models import UniDepthV1


# Path to the zip file
dataset_dir = "/mnt/kostas-graid/datasets/zuoxy/3dgp/data/"
horse_dir = "/home/zuoxy/cg/3dgp/data/lsun_horses_256_40k_with_depth"
horse_dir_new = "/home/zuoxy/cg/3dgp/data/lsun_horses"
imagenet_zip_path = '/home/zuoxy/cg/3dgp/data/imagenet_256_with_depth.zip'
imagenet_zip_path_new = '/home/zuoxy/cg/3dgp/data/imagenet_256_with_depth_new.zip'
horse_zip_path = "/home/zuoxy/cg/3dgp/data/lsun_horses_256_40k_with_depth_3dgp.zip"
horse_zip_path_new = "/home/zuoxy/cg/3dgp/data/lsun_horses_256_40k_with_depth_zoe.zip"
proj_dir = "/home/zuoxy/cg/3dgp/"
output_dir = proj_dir+"debug/"
img_filename = "/home/zuoxy/cg/3dgp/debug/horse.jpg"
depth_filename = "/home/zuoxy/cg/3dgp/debug/horse_depth.png"
# img_filename = "/home/zuoxy/cg/3dgp/data/lsun_horses_256_40k_with_depth/ffff98e287e1180201fe1512699ed3090edbe7af.jpg"
# depth_filename = "/home/zuoxy/cg/3dgp/data/lsun_horses_256_40k_with_depth/ffff98e287e1180201fe1512699ed3090edbe7af_depth.png"

class DepthPredictor():
    def __init__(
        self,
        proj_dir,
        output_dir,
        method = "zoe",
        metric = False
    ):
        self.method = method
        self.metric = metric
        self.proj_dir = proj_dir
        self.output_dir = output_dir

    def initialize_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #depth anything
        if self.method == "depth_anything":
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }
            encoder = 'vitl' # or 'vits', 'vitb'
            dataset = 'vkitti' # 'hypersim' for indoor model, 'vkitti' for outdoor model

            if self.metric:
                model = DepthAnythingV2(**{**model_configs[encoder]})
                model.load_state_dict(torch.load(f'submodules/depth_anything_v2/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
            else:
                model = DepthAnythingV2(**{**model_configs[encoder]})
                model.load_state_dict(torch.load(f'submodules/depth_anything_v2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
            model.eval()
            return model.to(device)
        
        # #unidepth
        # if self.method == 'unidepth':
        #     model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
        #     return model.to(device)

        #zoe depth
        if self.method == "zoe":
            repo = "isl-org/ZoeDepth"
            zoe_config = "ZoeD_N"
            model = torch.hub.load(repo, zoe_config, pretrained=True)        
            return model.to(device)

    def estimate_depth(self, model, img):
       
        if self.method == "depth_anything":
            depth = model.infer_image(np.array(img))
            depth = 255 - (depth - depth.min())/(depth.max()-depth.min())*255
            depth = Image.fromarray(depth.astype(np.uint8))
           
        if self.method == "zoe":
            depth = model.infer_pil(img)
            depth = (depth - depth.min())/(depth.max()-depth.min())*255
            depth = Image.fromarray(depth.astype(np.uint8))
           
        # if self.method == "unidepth":
        #     depth = model.infer(img)
        return depth
    
    def process_images(self, input_zip_path, output_zip_path):
        model = self.initialize_model()
        count = 0
        with zipfile.ZipFile(input_zip_path, 'r') as input_zip, zipfile.ZipFile(output_zip_path, 'w') as output_zip:
            for file_path in input_zip.namelist():
                # print(file_path)
                count += 1
                if file_path.endswith(('.png', '.jpg', 'jpeg')) and not "depth.png" in file_path:

                    with input_zip.open(file_path) as file:
                        img = Image.open(file).convert('RGB')
                               
                        filename = os.path.basename(file_path)
                        if img is None or img.size == 0:
                            continue
                        else: 
                            print(f"Loading {count}/{len(input_zip.namelist())}")
                        
                        file_dir = os.path.dirname(file_path)

                        #save rgb image
                        rgb_bytes_io = BytesIO()
                        img.save(rgb_bytes_io, format='JPEG')
                        # img.save("/home/zuoxy/cg/3dgp/debug/horse_dargb.jpg")
                        output_zip.writestr(f'{file_path}', rgb_bytes_io.getvalue())

                        #save depth image
                        depth_bytes_io = BytesIO()
                        depth = self.estimate_depth(model, img)
                        # depth.save("/home/zuoxy/cg/3dgp/debug/horse_dadepth.png")
                        # break
                        depth.save(depth_bytes_io, format='PNG')
                        basename, ext = os.path.splitext(filename)
                        output_zip.writestr(f'{file_dir}/{basename}_depth.png', depth_bytes_io.getvalue())                       


def main():
    predictor = DepthPredictor(proj_dir, output_dir)
    predictor.process_images(horse_zip_path, horse_zip_path_new)
    # predictor.process_images(imagenet_zip_path, imagenet_zip_path_new)

if __name__ == "__main__":
    main() 