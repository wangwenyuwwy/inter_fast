import struct

# record_size = struct.calcsize("=2H4B6d1?")
# with open('G:/bxl/STRANet-master/STRANet-master/gen_dataset/VVCSoftware_VTM-VTM-10.2-dataset/build/source/App/EncoderApp/bxl.txt', 'rb') as fd:
# # with open('G:/bxl/VVCSoftware_VTM-VTM-15.0_new/build\source/App/EncoderApp/kkk.txt', 'rb') as fd:
#   buffer = fd.read()
# n = 0
# while n * record_size < len(buffer):
#   tmpcost = [0, 0, 0, 0, 0, 0]
#   cux, cuy, cuh, cuw, split, last_split, tmpcost[0], tmpcost[1], tmpcost[2], tmpcost[3], tmpcost[4], tmpcost[5], channel = \
#     struct.unpack("=2H4B6d1?", buffer[n * record_size: (n + 1) * record_size])
#   print(cux, cuy, cuh, cuw, split, last_split, tmpcost[0], tmpcost[1], tmpcost[2], tmpcost[3], tmpcost[4], tmpcost[5], channel)
#   n += 1

s = "saved_from_server/run-10.23/compressed_37/0/file_1280x720.yuv"
print(s.split('_')[-1].split('.')[-2].split('x'))