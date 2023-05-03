import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import cv2
import os
from cnn import FaceCNN


def validate(model, dataset, batch_size):
    val_loader = data.DataLoader(dataset, batch_size)
    total = len(dataset)
    right = 0 
    # loss_function = nn.CrossEntropyLoss()
    # 防止梯度爆炸
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model.forward(images)
            right = right + (outputs.argmax(1)==labels).sum()  # 计数

    acc = right.item()/total
    return acc


class FaceDataset(data.Dataset):
    '''
    首先要做的是类的初始化。之前的image-emotion对照表已经创建完毕，
    在加载数据时需用到其中的信息。因此在初始化过程中，我们需要完成对image-emotion对照表中数据的读取工作。
    通过pandas库读取数据，随后将读取到的数据放入list或numpy中，方便后期索引。
    '''
    # 初始化
    def __init__(self, root):
        super(FaceDataset, self).__init__()
        self.root = root
        print(root)
        df_path = pd.read_csv(root + '/image_emotion.csv', header=None, usecols=[0])
        df_label = pd.read_csv(root + '/image_emotion.csv', header=None, usecols=[1])
        self.path = np.array(df_path)[:, 0]
        self.label = np.array(df_label)[:, 0]

    '''
    接着就要重写getitem()函数了，该函数的功能是加载数据。
    在前面的初始化部分，我们已经获取了所有图片的地址，在这个函数中，我们就要通过地址来读取数据。
    由于是读取图片数据，因此仍然借助opencv库。
    需要注意的是，之前可视化数据部分将像素值恢复为人脸图片并保存，得到的是3通道的灰色图（每个通道都完全一样），
    而在这里我们只需要用到单通道，因此在图片读取过程中，即使原图本来就是灰色的，但我们还是要加入参数从cv2.COLOR_BGR2GARY，
    保证读出来的数据是单通道的。读取出来之后，可以考虑进行一些基本的图像处理操作，
    如通过高斯模糊降噪、通过直方图均衡化来增强图像等（经试验证明，在本项目中，直方图均衡化并没有什么卵用，而高斯降噪甚至会降低正确率，可能是因为图片分辨率本来就较低，模糊后基本上什么都看不清了吧）。
    读出的数据是48X48的，而后续卷积神经网络中nn.Conv2d() API所接受的数据格式是(batch_size, channel, width, higth)，本次图片通道为1，因此我们要将48X48 reshape为1X48X48。
    '''

    # 读取某幅图片，item为索引号
    def __getitem__(self, item):
        face = cv2.imread(self.root + '/' + self.path[item])
        # 读取单通道灰度图
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # 高斯模糊
        # face_Gus = cv2.GaussianBlur(face_gray, (3,3), 0)
        # 直方图均衡化
        face_hist = cv2.equalizeHist(face_gray)
        # 像素值标准化
        face_normalized = face_hist.reshape(1, 48, 48) / 255.0 # 为与pytorch中卷积神经网络API的设计相适配，需reshape原图
        # 用于训练的数据需为tensor类型
        face_tensor = torch.from_numpy(face_normalized) # 将python中的numpy数据类型转化为pytorch中的tensor数据类型
        face_tensor = face_tensor.type('torch.FloatTensor') # 指定为'torch.FloatTensor'型，否则送进模型后会因数据类型不匹配而报错
        # face_tensor = face_tensor.cuda()
        label = self.label[item]
        label = torch.tensor(label)
        # label = label.cuda()
        return face_tensor, label


    '''
    最后就是重写len()函数获取数据集大小了。
    self.path中存储着所有的图片名，获取self.path第一维的大小，即为数据集的大小。
    '''
    # 获取数据集样本个数
    def __len__(self):
        return self.path.shape[0]
    


def train(val_dataset, batch_size):
    pretrained_model = "/Users/lanyiwei/data/model/0.pth"
    targeted_model=torch.load(pretrained_model)
    targeted_model.eval()
    acc_train = validate(targeted_model, val_dataset, batch_size)
    print('After epochs , the acc_val is : ', acc_train)

def main():
    # 数据集实例化(创建数据集)
    train_set = '/Users/lanyiwei/data/test_set'
    verify_set = '/Users/lanyiwei/data/verify_set'
    val_dataset = FaceDataset(root =verify_set)
    # val_dataset = FaceDataset(root =train_set)
    # 超参数可自行指定
    model = train( val_dataset, batch_size=128)

if __name__ == '__main__':
    main()