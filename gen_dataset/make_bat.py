import os

source_dir = r'G:\bxl\div_crop_400\0'
config_path = r'F:\LXJ\2023-12-25\encoder_intra_vtm.cfg'

source_basename = os.path.basename(source_dir)
yuv_files = sorted([f for f in os.listdir(source_dir) if f.lower().endswith('.yuv')])

commands = []
for yuv_file in yuv_files:
  base_name = os.path.splitext(yuv_file)[0]

  input_path = os.path.join(source_dir, yuv_file)
  bin_path = os.path.join('bin', 'qp37', f'{base_name}.bin')
  log_path = os.path.join('log', 'qp37', source_basename, f'{base_name}.txt')

  # 移除了所有路径周围的引号
  cmd = (
    f'EncoderApp_0.exe -c {config_path} '
    f'-i {input_path} '
    f'-fr 1 -f 1 -wdt 768 -hgt 512 -q 37 '
    f'-b {bin_path} > {log_path}'
  )
  commands.append(cmd)

with open('encode_commands.bat', 'w') as bat_file:
  bat_file.write('\n'.join(commands))

print(f'已生成 {len(commands)} 条编码命令到 encode_commands.bat 文件')