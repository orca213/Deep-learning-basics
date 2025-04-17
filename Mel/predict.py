import os
import torch
import torch.nn.functional as F
import librosa
import numpy as np
from cnn import GenreCNN_v2

# 🔧 설정
MODEL_PATH = "models/genre_cnn.pth"
AUDIO_PATH = "data"
GENRES = ['classical', 'jazz', 'metal', 'pop', 'hiphop', 'rock', 'blues', 'country', 'reggae', 'disco']

# 📂 오디오 파일 목록 가져오기
file_list = [f for f in os.listdir(AUDIO_PATH) if os.path.isfile(os.path.join(AUDIO_PATH, f))]

# 🔄 오디오 전처리 함수
def preprocess_audio(file_path, sr=22050, duration=30):
    y, sr = librosa.load(file_path, sr=sr, duration=duration)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel = librosa.power_to_db(mel)

    # Pad or truncate
    if log_mel.shape[1] < 660:
        pad = 660 - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode='constant')
    else:
        log_mel = log_mel[:, :660]

    tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 128, 660)
    return tensor

# 📦 모델 로드
model = GenreCNN_v2(num_classes=len(GENRES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# 🎧 오디오 불러오기 & 예측
for filename in file_list:
    print(f"\n🎧 Processing: {filename}")

    filepath = os.path.join(AUDIO_PATH, filename)
    input_tensor = preprocess_audio(filepath)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1).numpy().flatten()
        predicted_idx = np.argmax(probs)
        predicted_genre = GENRES[predicted_idx]

    # 출력
    print(f"🎵 Predicted Genre: {predicted_genre}")
    print("📊 Probabilities:")
    print(" | ".join([f"{genre:^10}" for genre in GENRES]))
    print(" | ".join([f"{prob:^10.4f}" for prob in probs]))
