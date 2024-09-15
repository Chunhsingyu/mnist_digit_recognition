mnist_digit_recognition/


├── README.md                 # 项目说明文档
│
├── data/                     # 存放MNIST数据集和训练完成的模型文件.pth
│
├── models/                   # 存放训练好的模型
│
├── notebooks/                # 存放jupyter notebooks
│
├── src/
│   ├── __init__.py
│   ├── dataset.py            # 数据加载
│   ├── model.py              # 构建模型
│   ├── train.py              # 模型训练
│   ├── evaluate.py           # 模型评估
│   ├── predict.py            # 预测任意手写数字
│
├── train.py                  # 主训练脚本
├── recognize.py               # 主评估脚本，用于检验模型是否正确识别sample.png
└── requirements.txt          # 依赖的库
