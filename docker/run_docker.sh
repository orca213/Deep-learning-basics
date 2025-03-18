#!/bin/bash

# 변수 설정
IMAGE_NAME="ubuntu_with_workspace"
CONTAINER_NAME="ubuntu_container"
WORKSPACE_PATH="C:\Users\박성제\문서\Deeplearning"

# 이미지가 이미 있는지 확인
if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "Docker image not found. Building the image..."
    docker build -t "$IMAGE_NAME" "$WORKSPACE_PATH/docker"
    if [ $? -ne 0 ]; then
        echo "Failed to build Docker image."
        exit 1
    fi
else
    echo "Docker image '$IMAGE_NAME' found. Skipping build."
fi

# 실행
echo "Running Docker container..."
docker run -it --rm --name "$CONTAINER_NAME" -v "$WORKSPACE_PATH:/workspace" "$IMAGE_NAME"
if [ $? -ne 0 ]; then
    echo "Failed to run Docker container."
    exit 1
fi

echo "Docker container exited successfully."
