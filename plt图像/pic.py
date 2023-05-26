import numpy as np
import matplotlib
import matplotlib; matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
#解决中文显示问题 ..未解决
# plt.rcParams["font.family"] = ["sans-serif"]
# plt.rcParams["font.sans-serif"] = ['SimHei']

# 读取存储为txt文件的数据
def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(',')

    return np.asfarray(data, float)


# 不同长度数据，统一为一个标准，倍乘x轴
def multiple_equal(x, y):
    x_len = len(x)
    y_len = len(y)
    times = x_len/y_len
    y_times = [i * times for i in y]
    return y_times


if __name__ == "__main__":

    # train_loss_path = 'deep leaning /plt图像/train_loss.txt'
    # train_acc_path = 'deep leaning /plt图像/train_acc.txt'
    train_loss_path=r'Z:\data\savedata\train_loss.txt'
    train_acc_path=r'Z:\data\savedata\train_acc.txt'


    y_train_loss = data_read(train_loss_path)
    y_train_acc = data_read(train_acc_path)

    x_train_loss = range(len(y_train_loss))
    x_train_acc = multiple_equal(x_train_loss, range(len(y_train_acc)))
    # figure 可以指定生成图像的长宽高 图像的编号 背景颜色 和边框颜色
    plt.figure(figsize=(9,6),facecolor='white')

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('iters')
    plt.ylabel('accuracy')

    plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
    # plt.plot(x_train_acc, y_train_acc,  color='red', linestyle="solid", label="train accuracy")
    plt.legend()

    # plt.title("损失函数的值", fontproperties="SimHei") 。。还是现实不了中文
    plt.title('acc')
    print(matplotlib.get_backend())
    plt.show()
