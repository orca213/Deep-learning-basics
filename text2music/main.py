import os
import torchaudio
from transformers import MusicgenForConditionalGeneration, AutoProcessor
from pathlib import Path
import torch

def main():
    # 🔹 1. 터미널에서 텍스트 입력
    prompt = input("🎤 Enter a text prompt for music generation: ").strip()
    if not prompt:
        print("❌ No prompt provided. Exiting.")
        return

    # 🔹 2. 저장 디렉토리 생성
    output_dir = Path("outputs/audio")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 🔹 3. 모델 저장 디렉토리 확인 및 로딩
    model_dir = Path("models/musicgen-small")
    if model_dir.exists():
        print("✅ Loading model and processor from local 'models/' directory.")
        model = MusicgenForConditionalGeneration.from_pretrained(model_dir)
        processor = AutoProcessor.from_pretrained(model_dir)
    else:
        print("⬇️ Downloading model and saving to 'models/musicgen-small'...")
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        model.save_pretrained(model_dir)
        processor.save_pretrained(model_dir)

    # 🔹 4. 텍스트 처리 및 음악 생성
    inputs = processor(text=[prompt], return_tensors="pt")
    with torch.no_grad():
        audio_values = model.generate(**inputs, max_new_tokens=1024)

    # 🔹 5. 오디오 저장
    filename = prompt.replace(" ", "_")[:50] + ".wav"
    filepath = output_dir / filename
    torchaudio.save(str(filepath), audio_values[0], 32000)
    print(f"🎵 Audio generated and saved to: {filepath}")

if __name__ == "__main__":
    main()
