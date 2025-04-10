#!/bin/bash

echo "📦 필요한 Python 패키지들을 설치합니다..."

apt-get update
apt-get install ffmpeg -y

pip install librosa

echo "✅ 설치 완료!"
