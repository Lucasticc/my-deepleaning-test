import matplotlib.pyplot as plt

# 从文本文件中读取数据
with open(r'Z:\data\savedata\acc_vall.txt', 'r') as f:
    data = []
    for line in f:
        # 将每行数据拆分成多个数字
        nums = line.strip().split(',')
        # 将每个数字转换为浮点数并添加到列表中
        print(nums)
        data += [float(num) for num in nums]

# 绘制折线图
plt.plot(data)

# 添加标题和轴标签
plt.title('Line Chart')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

# 显示图表
plt.show()