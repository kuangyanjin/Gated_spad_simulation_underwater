import numpy as np
import cv2
import os
 
 
def add_hazy(image, beta=0.1, brightness=0.5):
    '''
    :param image:   输入图像
    :param beta:    雾强
    :param brightness:  雾霾亮度
    :return:    雾图
    '''
    img_f = image.astype(np.float32) / 255.0
    row, col, chs = image.shape
    size = np.sqrt(max(row, col))  
    center = (row // 2, col // 2) 
    y, x = np.ogrid[:row, :col]
    dist = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    d = -0.04 * dist + size
    td = np.exp(-beta * d)
    img_f = img_f * td[..., np.newaxis] + brightness * (1 - td[..., np.newaxis])
    hazy_img = np.clip(img_f * 255, 0, 255).astype(np.uint8)
    return hazy_img

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
 
if __name__ == '__main__':
    path = r'/root/autodl-tmp/gated_spad_simulation_underwater/results/smu_img_67m_01.png'
    image = cv2.imread(path)
 
    image_fogv2 = add_hazy(image, beta=0.05, brightness=0.1)#0.05 0.8
    image_simulation = simulation_spad(image_fogv2)
    img_result = image_simulation[64]
    result = cv2.hconcat([image, image_fogv2])
    result = cv2.resize(result, None, fx=0.5, fy=0.5)
    #cv2.imwrite("/root/autodl-tmp/RESULT.png", result)
    cv2.imwrite("/root/autodl-tmp/gated_spad_simulation_underwater/results/add_haze_67m_0101.png", image_fogv2)
    cv2.imwrite("/root/autodl-tmp/gated_spad_simulation_underwater/results/add_haze_spad_64_67m_00501.png", img_result)
