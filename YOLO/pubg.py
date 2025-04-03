import torch
import cv2
from PIL import Image
import os

# 1. YOLOv5 사전학습 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 2. 타겟 클래스 설정
target_class = 'person'

# 3. 이미지 경로
img_path = 'data/pubg_screenshot.png'
img = Image.open(img_path)

# 4. 탐지 수행
results = model(img)

# 5. 결과 DataFrame
df = results.pandas().xyxy[0]
people = df[df['name'] == target_class]

# 6. OpenCV로 이미지 불러와서 박스 그리기
img_cv = cv2.imread(img_path)

for _, row in people.iterrows():
    x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
    confidence = row['confidence']
    
    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(img_cv, f"{target_class} {confidence:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# 7. 결과 저장
os.makedirs('result', exist_ok=True)
cv2.imwrite('result/pubg_result.jpg', img_cv)
print("🎉 사람 탐지 완료! 결과는 'result/pubg_result.jpg'에 저장되었어요.")
