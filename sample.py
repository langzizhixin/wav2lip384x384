"""数据集抽样"""
import os
import random

videos_dir = "佳音口播切片"
all_rel_paths = set()
for root, dirs, files in os.walk(videos_dir):
    for file in files:
        name, extension = os.path.splitext(file)
        if extension != '.mp4':
            continue
        file_path = os.path.join(root, file)
        file_rel_path = os.path.relpath(file_path, videos_dir)
        all_rel_paths.add(file_rel_path[:-4])

all_rel_paths_list = list(all_rel_paths)
random.shuffle(all_rel_paths_list)
# 计算索引以分割列表
total_samples = len(all_rel_paths_list)
train_idx = int(total_samples * 0.8)
test_idx = train_idx + int(total_samples * 0.1)
# 分割列表
train_paths = all_rel_paths_list[:train_idx]
test_paths = all_rel_paths_list[train_idx:test_idx]
val_paths = all_rel_paths_list[test_idx:]


# 将数据写入文件
def write_to_file(file_path, paths):
    with open(file_path, 'w') as file:
        for path in paths:
            file.write(f'{path}\n')


# 指定文件路径
train_file = 'filelists/train.txt'
test_file = 'filelists/test.txt'
val_file = 'filelists/val.txt'
# 写入文件
write_to_file(train_file, train_paths)
write_to_file(test_file, test_paths)
write_to_file(val_file, val_paths)
