import json

# 读取 JSON 文件
with open("G:/bxl/STRANet-master/STRANet-master/gen_dataset/0713_0/32/0001_768x512.json", "r") as f:
    data = json.load(f)

# 提取 prob 数组的第一个元素
prob_first = data.get("prob", [{}])[0]  # 处理空数组的情况

# 统计键值对个数
count = len(prob_first)
print(f"prob 第一个元素的键值对个数: {count}")