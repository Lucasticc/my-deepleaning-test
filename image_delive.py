import os
import shutil
import re

# 原文件夹路径
source_folder= r'Z:\data\face_image'

# 新文件夹路径
destination_folder= r'Z:\data\train_set'
def main():
    # 获取原文件夹中的所有文件
    files = os.listdir(source_folder)

    # 循环遍历每个文件，并剪切到新文件夹
    for file_name in files:
        numbers = re.findall('\d+', file_name)
        result = ''.join(numbers)
        result = int(result)
        print (result)
        if result in range(0,24001):
            # 构建原文件的完整路径
            source_file = os.path.join(source_folder, file_name)
            # 构建目标文件的完整路径
            destination_file = os.path.join(destination_folder, file_name)
            # 将文件剪切到新文件夹
            shutil.move(source_file, destination_file)
if __name__ == '__main__':
    main()