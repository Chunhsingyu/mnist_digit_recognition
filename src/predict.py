import torch
from PIL import Image
from torchvision import transforms
from src.model import SimpleCNN

def predict_image(image_path):
    model = SimpleCNN()
    model.load_state_dict(torch.load('./models/mnist_cnn.pth'))
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # 增加batch维度

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        print(f'Predicted digit: {predicted.item()}')

if __name__ == "__main__":
    image_path = input("Enter the path of the image: ")
    predict_image(image_path)
