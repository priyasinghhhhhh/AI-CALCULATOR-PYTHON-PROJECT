import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not opening!")
    exit()

print("Camera opened successfully!")

while True:
    success, frame = cap.read()

    if success:
        cv2.imshow("My Webcam", frame)
    else:
        print("Cannot read camera")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()