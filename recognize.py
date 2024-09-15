import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# 定义与训练时相同的模型结构
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # 输出尺寸: [batch_size, 32, 26, 26]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3) # 输出尺寸: [batch_size, 64, 24, 24]
        self.pool = nn.MaxPool2d(2, 2)               # 池化后尺寸: [batch_size, 64, 12, 12]
        self.fc1 = nn.Linear(64 * 12 * 12, 128)      # 输入大小与训练时一致
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))      # 输出尺寸: [batch_size, 32, 26, 26]
        x = torch.relu(self.conv2(x))      # 输出尺寸: [batch_size, 64, 24, 24]
        x = self.pool(x)                   # 输出尺寸: [batch_size, 64, 12, 12]
        x = x.view(-1, 64 * 12 * 12)       # 展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 加载模型
model = SimpleCNN()
model.load_state_dict(torch.load('./models/mnist_cnn.pth'))
model.eval()  # 设置为评估模式

# 定义图像预处理
transform = transforms.Compose([
    transforms.Grayscale(),            # 转换为灰度图像
    transforms.Resize((28, 28)),       # 调整图像大小为28x28
    transforms.ToTensor(),             # 转换为Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化
])

def predict_image(image_path):
    # 打开图像并进行预处理
    image = Image.open(image_path).convert('L')  # 确保图像为灰度图
    image = transform(image).unsqueeze(0)        # 添加批次维度 [1, 1, 28, 28]
    # 预测
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()

# 主函数
if __name__ == '__main__':
    image_path = './data/sample.png'  # 请替换为你的图像路径
    predicted_digit = predict_image(image_path)
    print(f'预测的数字是: {predicted_digit}')
