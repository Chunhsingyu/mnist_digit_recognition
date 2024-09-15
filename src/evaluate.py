import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.model import SimpleCNN
from src.dataset import get_dataloaders

def evaluate_model():
    _, test_loader = get_dataloaders(batch_size=1000)
    model = SimpleCNN()
    model.load_state_dict(torch.load('./models/mnist_cnn.pth'))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():  # 不计算梯度，加快速度
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = correct / total
    print(f'Accuracy of the model on the test dataset: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    evaluate_model()
