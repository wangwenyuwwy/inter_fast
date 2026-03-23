with open("G:/bxl/VVCSoftware_VTM-VTM-15.0_new/bin/vs16/msvc-19.29/x86_64/release/compressed_22/0/0001_768x512.txt", "rb") as f:
  data = f.read()  # 读取为字节流
  # print(data[:20].hex())  # 输出前20字节（如 b'\x89PNG\r\n\x1a\n...'）

try:
  text = data.decode("utf-8")  # 尝试UTF-8解码
  print(text)
except UnicodeDecodeError:
  text = data.decode("gbk")  # 尝试GBK解码
  print(text)