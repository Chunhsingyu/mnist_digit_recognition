import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
from src.model import SimpleCNN
from src.dataset import get_dataloaders

def train_model(epochs=10, batch_size=64, lr=0.01):
    train_loader, test_loader = get_dataloaders(batch_size)
    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()  # 梯度清零
            output = model(data)   # 前向传播
            loss = criterion(output, target)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}')
    
    torch.save(model.state_dict(), './models/mnist_cnn.pth')  # 保存模型

if __name__ == "__main__":
    train_model()
