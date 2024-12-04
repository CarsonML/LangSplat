import os
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm
import cv2

from dataclasses import dataclass, field
from typing import Tuple, Type
from copy import deepcopy

import torch
import torchvision
from torch import nn


def dino_slice(slice, dino_model):
    images = slice
    assert len(images.shape) == 4

    if images.shape[1] == 3:  # Check if it's a 3-channel image (B, C, H, W)
        images = images[:, [2, 1, 0], :, :]  # Swap Red and Blue channels for all images in the batch

    #Center crop to closest multiples of 14 for dino window size.
    height = 728
    width = 980
    
    transform = transforms.Compose([       

        transforms.CenterCrop(size=(height, width)),
    
        # Rescale by 1/255 
        transforms.Lambda(lambda x: x * 0.00392156862745098),
        
        # Normalize using mean and std - this is what Dino pretrained config uses for some reason
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    processed_images = torch.stack([transform(img.float()) for img in images])
    

    with torch.no_grad():
        outputs = dino_model(processed_images)
    
    last_hidden_states = outputs.last_hidden_state  # Shape: [batch_size, num_tokens, feature_dim]
    
    patch_features = last_hidden_states[:, 1:, :]  # Exclude the [CLS] token - not relevant for per-pixel info
    
    batch_size, num_patches, feature_dim = patch_features.shape
    
    patch_features_grid = patch_features.squeeze().reshape(-1, height//14, width//14, feature_dim)
    
    # Upsample the feature maps using bicubic interpolation to match the original image size (height x width)
    upsampled_features = F.interpolate(
        patch_features_grid.permute(0, 3, 1, 2),  # Permute to [batch_size, feature_dim, height_patches, width_patches]
        size=(height, width), 
        mode='bicubic',  # Bicubic interpolation
        align_corners=False
    )
    
    # The upsampled features tensor now has shape [batch_size, feature_dim, height, width]
    return upsampled_features

def create(image_list, data_list, save_folder):
    assert image_list is not None, "image_list must be provided to generate features"
    embed_size=768
    timer = 0
    model = Dinov2Model.from_pretrained("facebook/dinov2-base")
    height = 728
    width = 980
    

    batch_size = 16
    num_batches = image_list.size(0) // batch_size + 1  # Calculate number of batches

    img_embeds = torch.zeros_like(image_list)

    for i in tqdm(range(num_batches), desc="Processing batches", leave=True):
        # Get the current batch (slice the tensor)
        if (i + 1) * batch_size < len(image_list):
            start_index = i * batch_size
            end_index = (i + 1) * batch_size
        else:
            start_index = i * batch_size
            end_index = -1
        
        batch_input = image_list[start_index:end_index]
        dino_features = dino_slice(batch_input, model)
        img_embeds[start_index:end_index] = dino_features
        
        
    for i in range(img_embeds.shape[0]):
        save_path = os.path.join(save_folder, data_list[i].split('.')[0])
        curr = {
            'feature': img_embeds[i, :, :, :].reshape(embed_size, height * width).permute(1,0)
        }
        sava_numpy(save_path, curr)

def sava_numpy(save_path, data):
    save_path_s = save_path + '_s.npy'
    save_path_f = save_path + '_f.npy'
    np.save(save_path_f, data['feature'].numpy())

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    seed_num = 42
    seed_everything(seed_num)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--sam_ckpt_path', type=str, default="ckpts/sam_vit_h_4b8939.pth")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    dataset_path = args.dataset_path
    img_folder = os.path.join(dataset_path, 'images')
    data_list = os.listdir(img_folder)
    data_list.sort()

    img_list = []
    WARNED = False
    for data_path in data_list:
        image_path = os.path.join(img_folder, data_path)
        image = cv2.imread(image_path)

        orig_w, orig_h = image.shape[1], image.shape[0]
        
        ## not resizing, don't think I need the following
        # if args.resolution == -1:
        #     if orig_h > 1080:
        #         if not WARNED:
        #             print("[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n "
        #                 "If this is not desired, please explicitly specify '--resolution/-r' as 1")
        #             WARNED = True
        #         global_down = orig_h / 1080
        #     else:
        #         global_down = 1
        # else:
        #     global_down = orig_w / args.resolution
            
        # scale = float(global_down)
        # resolution = (int( orig_w  / scale), int(orig_h / scale))
        
        # image = cv2.resize(image, resolution)
        
        image = torch.from_numpy(image)
        img_list.append(image)
    images = [img_list[i].permute(2, 0, 1)[None, ...] for i in range(len(img_list))]
    imgs = torch.cat(images)

    save_folder = os.path.join(dataset_path, 'language_features')
    os.makedirs(save_folder, exist_ok=True)
    create(imgs, data_list, save_folder)
