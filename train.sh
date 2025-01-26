#!/bin/bash

#均由GPT生成

# 设置环境变量（如果有需要）
export CUDA_VISIBLE_DEVICES=0  # 使用GPU编号0（如果有多个GPU）

# 创建日志目录
mkdir -p logs
LOG_FILE="logs/train_$(date +'%Y%m%d_%H%M%S').log"

# 设置训练参数
BATCH_SIZE=1
EPOCHS=100
#LEARNING_RATE=0.001
DATA_PATH="./dataset"  # 数据集路径
WEIGHT_PATH="./params/unet.pth"  # 权重文件保存路径
SAVE_PATH="./train_image"  # 输出图像保存路径

# 创建保存模型的目录
mkdir -p $SAVE_PATH
mkdir -p $(dirname $WEIGHT_PATH)

# 启动训练脚本
echo "Starting training..." | tee -a $LOG_FILE
python train1.py --data_path $DATA_PATH --save_path $SAVE_PATH --weight_path $WEIGHT_PATH --batch_size $BATCH_SIZE --epochs $EPOCHS --learning_rate $LEARNING_RATE | tee -a $LOG_FILE

# 训练完成后保存模型
echo "Training completed. Model saved to $WEIGHT_PATH" | tee -a $LOG_FILE
