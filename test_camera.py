import cv2

for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        print(f"Camera index {i}: {'WORKS' if ret else ' Opens but cant read'}")
        cap.release()
    else:
        print(f"Camera index {i}:  Not found")