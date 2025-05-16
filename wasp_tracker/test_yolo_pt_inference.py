from ultralytics import YOLO
import cv2

# 모델 로드
model = YOLO("./models/best.pt")

# 이미지 로드
image = cv2.imread("sample2.jpg")

# BGR → RGB 변환
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 추론 실행
results = model(image_rgb)

# 결과 시각화 및 저장
# YOLO 결과 시각화 (RGB 형식)
annotated = results[0].plot()

# RGB → BGR 변환 후 저장
annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
cv2.imwrite("debug_output.jpg", annotated_bgr)


# 디텍션 요약 출력
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    print(f"[DETECTED] class={cls_id}, conf={conf:.2f}")
