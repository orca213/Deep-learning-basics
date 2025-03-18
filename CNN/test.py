from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from train import SimpleCNN
from tqdm import tqdm


if __name__ == "__main__":
    # 테스트 데이터셋 불러오기
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("cnn_model.pth", weights_only=True))
    print("model loaded")
    model.eval()

    # 테스트 정확도 계산
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    # 예시 5개 시각화
    examples = iter(test_loader)
    example_images, example_labels = next(examples)
    example_images, example_labels = example_images.to(device), example_labels.to(device)

    # 모델 예측
    with torch.no_grad():
        outputs = model(example_images)
        _, predicted = torch.max(outputs, 1)

    # 시각화
    fig, axes = plt.subplots(1, 10, figsize=(18, 3))
    for i in range(10):
        axes[i].imshow(example_images[i].cpu().squeeze(), cmap='gray')
        axes[i].set_title(f"Label: {example_labels[i].item()}\nPred: {predicted[i].item()}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()