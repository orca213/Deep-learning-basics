import os
import torch
import tarfile
import librosa
import numpy as np
from torch.utils.data import Dataset


def download_dataset():
    # Try to download GTZAN dataset using kagglehub if not found
    try:
        if not os.path.exists("genres"):
            import kagglehub
            print("📥 Downloading GTZAN dataset via kagglehub...")
            path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")
            print("✅ Dataset downloaded at:", path)

            # Check for genres.tar.gz and extract it
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith(".tar.gz"):
                        tar_path = os.path.join(root, file)
                        print(f"🗜️ Extracting {tar_path}...")
                        with tarfile.open(tar_path, "r:gz") as tar:
                            tar.extractall(path=os.path.dirname(tar_path))
                        print("✅ Extraction complete.")

            # Debugging: print folder structure
            print("📁 Searching for 'genres' folder...")
            found = False
            for root, dirs, files in os.walk(path):
                for d in dirs:
                    print("📂", os.path.join(root, d))  # print all folders
                    if d in ['genres', 'genres_original']:
                        genres_path = os.path.join(root, d)
                        found = True
                        break
                if found:
                    break

            if not found:
                raise FileNotFoundError("'genres' folder not found after extraction.")

            os.symlink(genres_path, "genres")
            print("✅ Linked 'genres/' to", genres_path)

    except Exception as e:
        print("❌ Failed to prepare dataset:", e)
        exit(1)


class GTZANDataset(Dataset):
    def __init__(self, data_dir, genres, duration=30, sr=22050):
        self.data = []
        self.labels = []
        self.genres = genres
        self.sr = sr

        for label, genre in enumerate(genres):
            genre_path = os.path.join(data_dir, genre)
            for file in os.listdir(genre_path):
                if file.endswith('.wav'):
                    file_path = os.path.join(genre_path, file)
                    try:
                        y, sr = librosa.load(file_path, sr=sr, duration=duration)
                        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                        log_mel = librosa.power_to_db(mel)

                        # Pad/truncate to fixed size
                        if log_mel.shape[1] < 660:
                            pad = 660 - log_mel.shape[1]
                            log_mel = np.pad(log_mel, ((0,0),(0,pad)), mode='constant')
                        else:
                            log_mel = log_mel[:, :660]

                        self.data.append(log_mel)
                        self.labels.append(label)
                    except Exception as e:
                        print(f"⚠️ Skipping {file_path}: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)  # (1, 128, 660)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return X, y
