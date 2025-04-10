# main.py
import torch
import cv2
import numpy as np
import mss
import pyautogui
import keyboard
import time

# 1. YOLOv5 모델 로드 (사전학습된 YOLOv5s)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.4  # confidence threshold

# 2. 감지 대상
TARGET_CLASSES = ['person', 'car', 'truck', 'bus', 'motorbike']

# 3. 화면 캡처 설정
monitor = {"top": 100, "left": 100, "width": 800, "height": 600}
sct = mss.mss()

def capture_screen():
    img = np.array(sct.grab(monitor))
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

def detect_objects(frame):
    results = model(frame)
    detections = results.pandas().xyxy[0]
    return detections[detections['name'].isin(TARGET_CLASSES)]

def decide_direction(detections, frame_width):
    left_count = 0
    right_count = 0
    for _, row in detections.iterrows():
        center_x = (row['xmin'] + row['xmax']) / 2
        if center_x < frame_width / 2:
            left_count += 1
        else:
            right_count += 1

    if left_count > right_count:
        return 'right'
    elif right_count > left_count:
        return 'left'
    else:
        return 'forward'

def press_key(direction):
    keyboard.release('a')
    keyboard.release('d')

    if direction == 'left':
        keyboard.press('a')
        print("⬅️ 왼쪽으로 회피")
    elif direction == 'right':
        keyboard.press('d')
        print("➡️ 오른쪽으로 회피")
    else:
        print("⬆️ 전진")

# 4. 루프 실행
print("🚗 자율주행 시작... ESC를 눌러 종료하세요.")
try:
    while True:
        frame = capture_screen()
        detections = detect_objects(frame)
        direction = decide_direction(detections, frame.shape[1])
        press_key(direction)

        # ESC 키 누르면 종료
        if keyboard.is_pressed('esc'):
            break

        time.sleep(0.1)  # 10 FPS 정도로 조절
finally:
    keyboard.release('a')
    keyboard.release('d')
    print("🛑 자율주행 종료.")
