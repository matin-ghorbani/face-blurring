import cv2 as cv
from face_detector import FaceDetector

cap = cv.VideoCapture(0)
detector = FaceDetector(min_detection_con=.75)

while True:
    _, frame = cap.read()
    frame = cv.resize(frame, (0, 0), None, .5, .5)

    frame, bboxs = detector.find_faces(frame, True)
    if bboxs:
        for i, bbox in enumerate(bboxs):
            x, y, w, h = bbox['bbox']
            if x < 0: x = 0
            if y < 0: y = 0
            frame_crop = frame[y:y + h, x:x + w]
            frame_blur = cv.blur(frame_crop, (60, 60))
            frame[y:y + h, x:x + w] = frame_blur

    cv.imshow('webcam', frame)
    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()
