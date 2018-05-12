## Transfer Learning based on ResNet18

- **模仿pytorch的教程写的**，主要实现Fine-Tuning（微调）

  用到的图片数据下载地址：https://download.pytorch.org/tutorial/hymenoptera_data.zip

  解压后，更名为data目录

- **实现功能**
  - 读取torchvision中保存的resnet18网络模型，设置预训练```pretrained=True```，修改最后一个全连接层为```my_resnet18.fc = nn.Linear(512, 2)```

  - 定义交叉熵损失函数```criterion = nn.CrossEntropyLoss()```

    定义随机梯度下降优化器`optimizer = optim.SGD(my_resnet18.parameters(), lr=0.001, momentum=0.9)`

    定义学习率每7步自动衰减`exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)`

  - 进行25轮次训练，每一轮都在训练集上训练，在验证集测试

  - 把25轮次中的最优模型参数保存下来`best_model_wts = copy.deepcopy(my_resnet18.state_dict())`

  - 最终模型读取最优参数`my_resnet18.load_state_dict(best_model_wts)`

- **注意**

  - 多卡运行可以平均使用每张卡，但是卡之间传输数据会大幅降低速度，除非模型非常大，否则不需要把模型放在多卡上运行

  - torchvision是pytorch基于计算机视觉的库，torchvision本身基于pytorch开发，

    里面有现成的数据集datasets：MNIST, cifar, imagenet, coco, pascal VOC等

    常用的预训练网络模型models：alexnet,vgg,inception,resnet等

    还有常用的变换transforms：RandomResizeCrop,ToTensor,Normalize等