import os
import numpy as np
from PIL import Image

# 定义源文件夹路径和目标文件夹路径
source_folder = 'G:/bxl/div/000'
target_folder = 'G:/bxl/div/000_crop'

# 如果目标文件夹不存在，则创建它
if not os.path.exists(target_folder):
  os.makedirs(target_folder)


def yuv2rgb(y, u, v):
  # 将YUV转换为RGB的函数（这里仅提供框架）
  r = y + 1.402 * (v - 128)
  g = y - 0.34414 * (u - 128) - 0.71414 * (v - 128)
  b = y + 1.772 * (u - 128)
  return r, g, b

def crop_center_yuv(file_path, output_folder, target_width=768, target_height=512):
    # 提取宽度和高度信息
    base_name = os.path.basename(file_path)
    name_parts = base_name.split('_')
    width, height = [int(x) for x in name_parts[-1].split('.')[0].split('x')]

    # 确保目标尺寸不超过原图尺寸
    if target_width > width or target_height > height:
      print("目标尺寸超过原图尺寸")
      return

    # 计算裁剪边界
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    # 读取YUV数据
    with open(file_path, 'rb') as f:
      yuv = np.frombuffer(f.read(), dtype=np.uint8)

      # 假设YUV420p格式，计算UV平面的起始位置
      y_size = width * height
      uv_size = y_size // 4

      y_plane = yuv[:y_size].reshape(height, width)
      u_plane = yuv[y_size:y_size + uv_size].reshape(height // 2, width // 2)
      v_plane = yuv[y_size + uv_size:].reshape(height // 2, width // 2)

    # 裁剪YUV数据
    y_cropped = y_plane[top:bottom, left:right]
    u_cropped = u_plane[top // 2:bottom // 2, left // 2:right // 2]
    v_cropped = v_plane[top // 2:bottom // 2, left // 2:right // 2]

    # 合并裁剪后的YUV数据
    yuv_cropped = np.concatenate([y_cropped.flatten(),
                                  u_cropped.flatten(),
                                  v_cropped.flatten()])

    # 动态生成新的文件名，保留原始命名规则
    new_filename = '_'.join(name_parts[:-1]) + f'_{target_width}x{target_height}.yuv'
    output_path = os.path.join(output_folder, new_filename)

    # 将裁剪后的YUV数据写入新文件
    with open(output_path, 'wb') as f:
      f.write(yuv_cropped.tobytes())


# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
  if filename.endswith('.yuv'):
    file_path = os.path.join(source_folder, filename)
    crop_center_yuv(file_path, target_folder)

print("处理完成")