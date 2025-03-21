import cv2
import torch
import numpy as np
import os
from kornia.filters import gaussian_blur2d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_depth_map(image, d0):
    height, width = image.shape[1], image.shape[2]
    return torch.full((height, width), d0, dtype=torch.float32, device=device)

def underwater_scattering_model(image, depth_map, c=0.2, kb=0.5, sigma_scale=0.001):
    t = torch.exp(-c * depth_map)
    direct_light = image * t.unsqueeze(0)
    
    backscatter = kb * (1 - torch.exp(-2 * c * depth_map))
    backscatter_layer = backscatter.unsqueeze(0).repeat(3, 1, 1)
    
    sigma = sigma_scale * depth_map.max()
    kernel_size = int(6 * sigma + 1)
    kernel_size = max(kernel_size, 3) | 1
    
    forward_scatter = gaussian_blur2d(
        direct_light.unsqueeze(0), 
        (kernel_size, kernel_size), 
        (sigma.item(), sigma.item())
    ).squeeze(0)
    
    output = direct_light + backscatter_layer + forward_scatter
    return torch.clamp(output, 0, 255).to(torch.uint8)

def simulation_spad(image, scale_factor=100.0, total_frames=256, frames=[1, 4, 16, 64, 256]):
    y_channel = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
    scaled_y = y_channel / scale_factor
    
    photon_counts = torch.poisson(
        scaled_y.unsqueeze(0).expand(total_frames, *scaled_y.shape)
    ).permute(1, 2, 0).unsqueeze(2).expand(-1, -1, 3, -1)
    
    results = {}
    for fil in frames:
        cumulated = torch.mean((photon_counts[:, :, :, :fil] >= 1).float(), dim=3)
        results[fil] = (cumulated * 255).clamp(0, 255).to(torch.uint8)
    return results

if __name__ == '__main__':
    input_folder = '/root/autodl-tmp/ground_data/10m'
    output_folder = '/root/autodl-tmp/smu_noisy_data/10m'
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        input_path = os.path.join(input_folder, filename)
        image_bgr = cv2.imread(input_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float().to(device)
        
        depth_map = get_depth_map(image_tensor, d0=10.0)
        image_smu = underwater_scattering_model(image_tensor, depth_map, c=0.1, kb=0.5, sigma_scale=0.1)
        results = simulation_spad(image_smu)
        
        output_path = os.path.join(output_folder, filename)
        if 64 in results:
            # 直接获取HWC格式的numpy数组
            result_np = results[64].cpu().numpy().astype(np.uint8)
            
            # 检查维度并转换颜色空间
            if result_np.shape[-1] != 3:
                result_np = np.transpose(result_np, (1, 2, 0))  # 转换为HWC
                
            result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, result_bgr)