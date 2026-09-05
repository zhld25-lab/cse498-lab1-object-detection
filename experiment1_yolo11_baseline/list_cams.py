import cv2
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ok, frame = cap.read()
        if ok:
            print(f"index {i}: OK, {frame.shape[1]}x{frame.shape[0]}")
            cv2.imshow(f"camera {i}", frame)
            cv2.waitKey(1500)
            cv2.destroyAllWindows()
    cap.release()