from torchvision import datasets, transforms
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # MNIST 데이터셋 불러오기 (훈련 데이터)
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    print(f"훈련 데이터셋 크기: {len(train_dataset)}")

    # 첫 번째 이미지 5개 시각화
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))

    for i in range(5):
        image, label = train_dataset[i]
        axes[i].imshow(image.squeeze(), cmap='gray')  # 이미지를 시각화 (채널 1을 제거)
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')  # 축 숨기기

    plt.show()