import os
import cv2
import numpy as np
import torch
from net import UNet  # 假设你的模型在 net.py 中定义
from utils import keep_image_size_open_rgb  # 保证这两个函数在 utils.py 中定义
from torchvision import transforms
from PIL import Image

# 定义数据转换操作
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为tensor
])

# 初始化网络模型
net = UNet(4)  # 假设4个类别，不需要.cuda()，默认为CPU

# 加载预训练的权重
weights = 'params/unet.pth'
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))  # 强制加载到CPU
    print('successfully loaded weights')
else:
    print('no weights loaded')

# 获取输入图片路径
_input = input('Please input JPEGImages path: ')

# 读取输入图像
img = keep_image_size_open_rgb(_input)  # 假设该函数读取并返回RGB图像

# 转换为tensor并确保在CPU上
img_data = transform(img)  # 默认转换为tensor，保存在CPU
img_data = img_data.unsqueeze(dim=0)  # 增加batch维度

# 设定模型为评估模式
net.eval()

# 模型预测
out = net(img_data)  # 在CPU上执行

# 获取最大值的类别索引
out = torch.argmax(out, dim=1)  # 输出类别索引
out = torch.squeeze(out, dim=0)  # 去掉batch维度
out = out.numpy()  # 将torch tensor转换为numpy数组，形状为(256, 256)

# 打印输出类别索引的唯一值
print(set(out.reshape(-1).tolist()))

# 将输入图像转换为numpy数组（原始图像）
img = np.array(img)

# 创建一个空的图像来存储结果
result_img = np.zeros_like(img)

# 背景区域（例如标签为0的部分）保留原始图像
# 对于其他区域，根据预测的类别来修改
for c in range(3):  # 对于每个颜色通道
    result_img[:, :, c] = np.where(out == 0, img[:, :, c], 0)  # 保留背景颜色

# 对于非背景区域，用模型预测的类别色填充
# 此处假设你有一个字典，将类别映射为颜色
class_colors = {
    1: [255, 0, 0],  # 红色
    2: [0, 255, 0],  # 绿色
    3: [0, 0, 255],  # 蓝色
}

for label, color in class_colors.items():
    mask = (out == label)  # 找到属于该类别的区域
    for c in range(3):  # 对每个颜色通道
        result_img[:, :, c] = np.where(mask, color[c], result_img[:, :, c])

# 保存并显示结果
cv2.imwrite('result/result.png', result_img)  # 保存结果图片
cv2.imshow('out', result_img)  # 显示图片
cv2.waitKey(0)
cv2.destroyAllWindows()
