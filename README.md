# 使用说明

###  环境搭载：（具体依赖库在requirements.txt中）
在赋予脚本执行权限后：
chmod +x setup_environment.sh    

运行脚本 setup_environment.sh 即可设置环境：
./setup_environment.sh

###  文件运行：
1.数据集原图存放至 dataset/JEPGImages ， mask存放至 dataset/SegmentationClass

2.运行 train.py ，可在文件 train_image中看到训练效果图

3.权重文件保存在 params 文件中

4.运行test.py ,输入要处理的图片地址进行测试即可

###  脚本运行
1.在train.sh中设置参数来进行模型的训练，初始参数为：

BATCH_SIZE=1

EPOCHS=100

DATA_PATH="./dataset"  # 数据集路径

WEIGHT_PATH="./params/unet.pth"  # 预训练模型路径

SAVE_PATH="./train_image"  # 保存训练图像路径

2.赋予脚本执行权限后：
chmod +x train.sh

执行训练脚本即可：
./train.sh
