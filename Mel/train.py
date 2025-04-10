import os
import tarfile
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from dataset import GTZANDataset, download_dataset
from cnn import GenreCNN

# Download the dataset if it doesn't exist
download_dataset()
    
# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 100
LR = 0.001
GENRES = ['classical', 'jazz', 'metal', 'pop', 'hiphop', 'rock', 'blues', 'country', 'reggae', 'disco']

if __name__ == "__main__":    
    # Load dataset
    dataset = GTZANDataset('genres/', GENRES)
    train_size = int(0.8 * len(dataset))
    train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    # Model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GenreCNN(num_classes=len(GENRES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Train
    epoch_pbar = tqdm(total=EPOCHS, desc="Training", unit="epoch", leave=False)
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        epoch_pbar.set_postfix(loss=f"{avg_loss:.4f}")
        epoch_pbar.update(1)
    print("Training complete!")

    # Evaluate
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            preds = outputs.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    print("="*50)
    print(f"Validation Accuracy: {correct / total:.2%}")
    
    # Save model
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(model.state_dict(), "models/genre_cnn.pth")
