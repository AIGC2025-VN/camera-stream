from ultralytics import YOLO
import cv2
import time  # Thêm thư viện để hỗ trợ kết nối lại camera

# Khởi tạo mô hình YOLOv8 (dùng phiên bản nhẹ yolov8n)
model = YOLO('yolov8n.pt')

# Địa chỉ camera IP
cctvcamera = "rtsp://admin:L2AB4B8A@192.168.1.106:554/cam/realmonitor?channel=1&subtype=0"
cap = cv2.VideoCapture(cctvcamera)

# Kiểm tra camera có mở thành công không
if not cap.isOpened():
    print("❌ Không thể mở camera. Vui lòng kiểm tra lại.")
    exit()

print("✅ Camera đã được mở thành công.")

cv2.namedWindow("YOLOv8 Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Detection", 800, 600)

# Bộ đếm để quản lý kết nối lại
reconnect_attempts = 0
max_reconnect_attempts = 5

while True:
    ret, frame = cap.read()

    # Xử lý mất kết nối camera
    if not ret:
        print(f"⚠️ Mất kết nối với camera. Đang thử kết nối lại ({reconnect_attempts + 1}/{max_reconnect_attempts})...")
        reconnect_attempts += 1

        # Ngủ 2 giây trước khi thử kết nối lại
        time.sleep(2)

        # Thử kết nối lại
        cap = cv2.VideoCapture(cctvcamera)
        if reconnect_attempts >= max_reconnect_attempts:
            print("❗ Không thể kết nối lại sau nhiều lần thử. Đang thoát...")
            break
        continue  # Quay lại vòng lặp để thử lại

    # Reset bộ đếm khi kết nối thành công
    reconnect_attempts = 0

    # Chạy dự đoán trên frame
    results = model(frame)

    # Tạo bản sao của khung hình gốc
    annotated_frame = frame.copy()

    # Vẽ khung đối tượng người lên ảnh
    for result in results[0].boxes:
        if model.names[int(result.cls)] == 'person':  
            annotated_frame = results[0].plot()

    # Hiển thị kết quả
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🔄 Đang thoát...")
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
