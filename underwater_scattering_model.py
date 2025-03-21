import cv2
import numpy as np
import os

def get_depth_map(image, d0):
    # 已知拍摄距离d0（单位：米）
    height, width = image.shape[:2]
    depth_map = np.full((height, width), d0, dtype=np.float32)
    return depth_map


def underwater_scattering_model(image, depth_map, c=0.2, kb=0.5, sigma_scale=0.1):
    """
    模拟水下散射效应
    :param image: 输入图像（BGR格式）
    :param depth_map: 深度图（归一化到[0,1]）
    :param c: 衰减系数（默认0.2）
    :param kb: 后向散射强度（默认0.5）
    :param sigma_scale: 前向散射模糊系数（默认0.1）
    :return: 合成的水下图像
    """
    # 计算透射率
    t = np.exp(-c * depth_map)
    
    # 直接光衰减
    direct_light = image * t[:, :, np.newaxis]
    
    # 后向散射层
    backscatter = kb * (1 - np.exp(-2 * c * depth_map))
    backscatter_layer = np.zeros_like(image)
    for ch in range(3):
        backscatter_layer[:, :, ch] = backscatter
    
    # 前向散射模糊
    sigma = sigma_scale * depth_map.max() * image.shape[1]
    forward_scatter = np.zeros_like(image)
    for ch in range(3):
        forward_scatter[:, :, ch] = cv2.GaussianBlur(direct_light[:, :, ch], (0,0), sigmaX=sigma)
    
    # 合成图像
    output = direct_light + backscatter_layer + forward_scatter
    return np.clip(output, 0, 255).astype(np.uint8)

def simulation_spad(image, scale_factor=100.0, total_frames=256, frames=[1, 4, 16, 64, 256]):
    """
    模拟SPAD相机低光成像（统一亮度通道噪声），生成多帧累积重建图像。
    
    参数:
        image (np.ndarray): 输入图像（HWC格式，值域0-255）。
        scale_factor (float): 控制光照强度的缩放因子（默认100.0）。
        total_frames (int): 总模拟帧数（默认256）。
        frames (list): 累积帧数列表（默认[1,4,16,64,256]）。
    
    返回:
        dict: 键为累积帧数，值为重建图像（uint8格式）。
    """
    # 转换为YUV并提取亮度通道（Y）
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y_channel = yuv_image[:, :, 0]  # 亮度通道
    height, width = y_channel.shape
    
    # 调整光照强度并生成泊松噪声（仅亮度通道）
    scaled_y = y_channel.astype(np.float32) / scale_factor
    photon_counts = np.random.poisson(
        scaled_y.flatten(), 
        size=(total_frames, height * width)
    ).T  # 形状: (H*W, total_frames)
    
    # 重塑为 (H, W, 1, total_frames) 并复制到三通道
    photon_counts = photon_counts.reshape(height, width, 1, total_frames)
    photon_counts = np.repeat(photon_counts, 3, axis=2)  # 形状 (H, W, 3, total_frames)
    
    # 二值化处理（光子到达为1，否则为0）
    b_counts = (photon_counts >= 1).astype(np.uint8)
    
    # 多帧累积重建
    results = {}
    for fil in frames:
        recon_image = np.mean(b_counts[:, :, :, :fil], axis=3)
        recon_image = (recon_image * 255).clip(0, 255).astype(np.uint8)
        results[fil] = recon_image
    
    return results


# if __name__ == '__main__':
#     path = r'/root/autodl-tmp/gated_spad_simulation_underwater/data/20m.png'
#     image = cv2.imread(path)

#     depth_map = get_depth_map(image , d0=30.0)
#     image_smu = underwater_scattering_model(image,depth_map,c=0.1, kb=0.5, sigma_scale=0.1)
#     image_simulation = simulation_spad(image_smu)
#     img_result = image_simulation[64]
#     cv2.imwrite("/root/autodl-tmp/DMID/data/CC/Noisy/30m.png", img_result)

if __name__ == '__main__':
    input_folder = r'/root/autodl-tmp/ground_data/10m'  # 输入图片文件夹
    output_folder = r'/root/autodl-tmp/smu_noisy_data/10m'  # 输出结果文件夹
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_folder, exist_ok=True)
    
    # 遍历输入文件夹中的所有图片文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # 处理单张图片
            image = cv2.imread(input_path)
            depth_map = get_depth_map(image, d0=10.0)
            image_smu = underwater_scattering_model(image, depth_map, c=0.1, kb=0.5, sigma_scale=0.1)
            #results = simulation_spad(image_smu)
            cv2.imwrite(output_path,image_smu)

            
            # 保存64帧累积结果
            if 64 in results:
                cv2.imwrite(output_path, results[64])
            else:
                print(f"Warning: No 64-frame result for {filename}")