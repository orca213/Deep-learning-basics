from torchvision import transforms
import torch
import numpy as np
from PIL import Image
from train import SimpleCNN

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((28, 28)),  # Resize to 28x28
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize
    ])

    # 이미지 로드
    image_path = "data/HAND/image.png"
    image = Image.open(image_path)
    image = Image.fromarray(255 - np.array(image))  # Invert the grayscale image
    input_image = transform(image).unsqueeze(0)  # Add batch dimension

    # 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("cnn_model.pth", weights_only=True))
    print("model loaded")
    model.eval()

    # 모델 예측
    input_image = input_image.to(device)
    with torch.no_grad():
        output = model(input_image)
        _, predicted = torch.max(output, 1)

    # 시각화
    plt.imshow(image, cmap='gray')
    plt.title(f"Predicted: {predicted.item()}")
    plt.axis('off')
    plt.show()