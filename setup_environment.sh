#!/bin/bash

#均由GPT生成

# 1. 创建新的 Conda 虚拟环境
echo "创建虚拟环境：myenv"
conda create -n myenv python=3.8 -y

# 激活虚拟环境
echo "激活虚拟环境：myenv"
conda activate myenv

# 2. 安装 PyTorch 以及其他依赖
echo "安装 PyTorch 和其他依赖"
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -y

# 安装其他常用库
conda install numpy matplotlib pandas scikit-learn opencv -y

# 安装必要的工具库
pip install -r requirements.txt

# 3. 检查 CUDA 设备是否可用
echo "检查 CUDA 设备"
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

# 4. 配置数据集路径（可选）
echo "配置数据集路径"
# 假设你已经下载了数据集，设置环境变量 DATA_PATH
export DATA_PATH=/path/to/your/dataset

# 5. 提示用户关于进一步的操作
echo "环境配置完毕！你现在可以运行你的训练脚本：train.sh"

