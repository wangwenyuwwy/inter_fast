import re


def replace_dimensions_in_bat(input_bat_file_path, output_bat_file_path, old_width=1984, old_height=1344, new_width=768,
                              new_height=512):
  # 定义旧的和新的文件路径模式
  old_path_pattern = re.compile(r'E:\\DGC\\div2k_ori\\(\d+)_(\d+)x(\d+)\.yuv')
  new_path_template = r'G:\bxl\div_crop\0\{file_number}_{new_width}x{new_height}.yuv'

  # 用于匹配 -wdt 和 -hgt 参数的正则表达式模式
  param_pattern = re.compile(r'(-wdt\s+\d+|-hgt\s+\d+)')

  # 打开并读取.bat文件内容
  with open(input_bat_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

  modified_lines = []
  for line in lines:
    # 替换 -wdt 和 -hgt 参数后的值
    line = param_pattern.sub(lambda m: f'-wdt {new_width}' if 'wdt' in m.group() else f'-hgt {new_height}', line)

    # 替换文件路径中的宽度和高度，并保持文件名序号不变
    def replace_filepath(match):
      file_number = match.group(1)
      return new_path_template.format(file_number=file_number, new_width=new_width, new_height=new_height)

    line = old_path_pattern.sub(replace_filepath, line)

    # 替换输出重定向部分中的宽度和高度
    redirect_pattern = re.compile(r'(>\s*\S*\\qp\d+\\\d+_\d+x\d+\.\w+)')
    old_dimension = f'{old_width}x{old_height}'
    new_dimension = f'{new_width}x{new_height}'
    line = redirect_pattern.sub(lambda m: m.group().replace(old_dimension, new_dimension), line)

    modified_lines.append(line)

  # 将修改后的内容写入新文件
  with open(output_bat_file_path, 'w', encoding='utf-8') as file:
    file.writelines(modified_lines)

  print(f"已成功创建并修改为新文件: {output_bat_file_path}")


# 使用函数来更新.bat文件中的尺寸，并保存到新文件中
input_bat_file_path = 'G:/bxl/VVCSoftware_VTM-VTM-15.0_new/bin/vs16/msvc-19.29/x86_64/release/trainset_vtm.bat'
output_bat_file_path = 'G:/bxl/VVCSoftware_VTM-VTM-15.0_new/bin/vs16/msvc-19.29/x86_64/release/trainset_vtm_new_200.bat'
replace_dimensions_in_bat(input_bat_file_path, output_bat_file_path)

print("修改完成")