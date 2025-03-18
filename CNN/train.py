from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


# 간단한 CNN 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # (1, 28, 28) -> (32, 28, 28)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # (32, 28, 28) -> (64, 28, 28)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)  # (64, 14, 14) -> Linear 입력 크기 수정
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)  # 풀링 크기 2x2
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))  # Conv1 -> (64, 28, 28)
        x = self.pool(self.relu(self.conv2(x)))  # Conv2 + Pooling -> (64, 14, 14)
        x = x.view(-1, 64 * 14 * 14)  # Flatten: 배치 크기 유지, 채널 64, 크기 14x14
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

if __name__ == "__main__":
    # MNIST 데이터셋 불러오기 (훈련 데이터)
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 모델, 손실 함수, 옵티마이저 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 모델 학습
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # 모델 저장
    torch.save(model.state_dict(), "cnn_model.pth")
    print("Model saved as cnn_model.pth")