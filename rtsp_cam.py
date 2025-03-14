import cv2

wificam = "rtsp://admin:L2AB4B8A@192.168.1.106:554/cam/realmonitor?channel=1&subtype=0"
cap = cv2.VideoCapture(wificam)
while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.resizeWindow('frame', 800, 600)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break