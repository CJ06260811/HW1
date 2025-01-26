# 使用说明
（由于.sh文件在Windows系统上难以运行，且.sh脚本并未经过测试，所以建议使用Addings运行流程）

###  环境搭载：（具体依赖库在requirements.txt中，也可使用cmd创建环境安装）
在赋予脚本执行权限后：
chmod +x setup_environment.sh    

运行脚本 setup_environment.sh 即可设置环境：
./setup_environment.sh

###  Addings-文件具体运行流程：
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

3.赋予脚本执行权限后：
chmod +x test.sh

执行测试脚本即可：
./test.sh test_image.jpg 

其中，test_image.jpg为图像路径
