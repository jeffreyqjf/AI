# AI
## 详情请见开发文档
本项目采用的数据集是MNIST数据集，即识别手写的数字0~9
### model framework
第一个model的框架如下(详细代码见networks.py)(相比之下为使用卷积的浅层模型)
卷积: 1 channels --> 64 channels kernel_size = 5
激活函数： Tanh
最大池化： 核大小为2，图像缩小一半
卷积: 64 channels --> 32 channels kernel_size = 3
激活函数： Tanh
最大池化： 核大小为2，图像缩小一半
卷积: 32 channels --> 16 channels kernel_size = 3
激活函数： Tanh
将张量展开成全连接层
先经过一次Linear： --> 128
激活函数： Tanh
经过一次Linear: --> 10

经过交叉熵损失
（相当于经过softmax --> 概率
再经过NLL）
反向传播 batch_size = 64

得到的model的权重保存在my_pt文件夹下面。命名规范为 {迭代次数}_model.pt