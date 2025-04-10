import torch
from PIL import Image
import cv2
import os

# 모델 로드 (사전학습된 YOLOv5s 사용)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 클래스 이름 (YOLOv5s는 'cup' 클래스를 포함하고 있음)
target_classes = ['cup']

# 이미지 불러오기
img_path = 'data/image.png'
img = Image.open(img_path)

# 객체 탐지 수행
results = model(img)

# 결과 필터링 (cup만 남기기)
df = results.pandas().xyxy[0]
cups = df[df['name'].isin(target_classes)]

# 이미지에 박스 그리기
img_cv = cv2.imread(img_path)

for _, row in cups.iterrows():
    x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
    label = row['name']
    confidence = row['confidence']
    
    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img_cv, f"{label} {confidence:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# 결과 저장 및 확인
os.makedirs('result', exist_ok=True)
cv2.imwrite('result/cafe_result.jpg', img_cv)
print("감지 완료! 'result/cafe_result.jpg' 파일을 확인하세요.")
