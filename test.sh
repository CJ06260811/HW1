#!/bin/bash

#均由GPT生成

# 设置Python环境，确保运行的是正确的环境
# 如果你有conda环境，可以激活特定环境
# conda activate your_environment_name

# 设置模型文件路径
MODEL_PATH="params/unet.pth"

# 设置输入和输出文件夹路径
INPUT_IMAGE_PATH=$1  # 输入图像路径，通过命令行参数传递
OUTPUT_PATH="result"

# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
  echo "Model weights not found at $MODEL_PATH"
  exit 1
fi

# 创建输出目录（如果不存在）
mkdir -p "$OUTPUT_PATH"

# 运行Python脚本进行推理
python test_model.py "$INPUT_IMAGE_PATH" "$OUTPUT_PATH"

# 输出结果
echo "Testing completed. Results are saved in $OUTPUT_PATH."
