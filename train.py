import os
import argparse
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from data import MyDataset  # 假设你的数据集在 data.py 中
from net import UNet  # 假设你的网络模型在 net.py 中
from torchvision.utils import save_image

# 设置命令行参数（由GPT生成）
def parse_args():
    parser = argparse.ArgumentParser(description="Train a UNet model")
    parser.add_argument('--data_path', type=str, default=r'D:\PY\dataset', help='Path to dataset')
    parser.add_argument('--save_path', type=str, default='train_image', help='Path to save images during training')
    parser.add_argument('--weight_path', type=str, default='params/unet.pth', help='Path to save/load model weights')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train the model')
#    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    return parser.parse_args()

def main():
    args = parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据加载
    data_loader = DataLoader(MyDataset(args.data_path), batch_size=args.batch_size, shuffle=True)

    # 初始化模型
    num_classes = 3 + 1  # +1 for background
    net = UNet(num_classes).to(device)

    # 加载预训练权重
    if os.path.exists(args.weight_path):
        net.load_state_dict(torch.load(args.weight_path))
        print('Successfully loaded weight!')
    else:
        print('No pre-trained weight found, training from scratch.')

    # 设置优化器和损失函数
    opt = optim.Adam(net.parameters(), lr=args.learning_rate)
    loss_fun = nn.CrossEntropyLoss()

    # 训练循环
    epoch = 1
    while epoch <= args.epochs:
        for i, (image, segment_image) in enumerate(data_loader):
            image, segment_image = image.to(device), segment_image.to(device)
            out_image = net(image)
            train_loss = loss_fun(out_image, segment_image.long())
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            if i % 1 == 0:
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')

            # 保存训练图像
            _image = image[0]
            _segment_image = torch.unsqueeze(segment_image[0], 0) * 255
            _out_image = torch.argmax(out_image[0], dim=0).unsqueeze(0) * 255

            img = torch.stack([_segment_image, _out_image], dim=0)
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            save_image(img, os.path.join(args.save_path, f'{epoch}_{i}.png'))

        # 每5个epoch保存一次模型
        if epoch % 5 == 0:
            torch.save(net.state_dict(), args.weight_path)
            print(f'Model saved to {args.weight_path}')

        epoch += 1

if __name__ == "__main__":
    main()
