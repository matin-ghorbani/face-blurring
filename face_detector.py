import cv2 as cv
from mediapipe.python.solutions.face_detection import FaceDetection


class FaceDetector:
    """
    Find faces in realtime using the light weight model provided in the mediapipe
    library.
    """

    def __init__(self, min_detection_con=0.5, model_selection=0):
        self.results = None
        self.min_detection_con = min_detection_con
        self.model_selection = model_selection
        self.face_detection = FaceDetection(min_detection_confidence=self.min_detection_con,
                                            model_selection=self.model_selection)

    def find_faces(self, img, draw=True):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.face_detection.process(img_rgb)
        bboxs = []
        if self.results.detections:
            for i, detection in enumerate(self.results.detections):
                if detection.score[0] > self.min_detection_con:
                    bbox_c = detection.location_data.relative_bounding_box
                    ih, iw, ic = img.shape
                    bbox = int(bbox_c.xmin * iw), int(bbox_c.ymin * ih), \
                        int(bbox_c.width * iw), int(bbox_c.height * ih)
                    cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)
                    bbox_info = {"id": i, "bbox": bbox, "score": detection.score, "center": (cx, cy)}
                    bboxs.append(bbox_info)
                    if draw:
                        img = cv.rectangle(img, bbox, (0, 0, 255), 2)
                        cv.putText(img, f'{int(detection.score[0] * 100)}%',
                                   (bbox[0], bbox[1] - 20), cv.FONT_HERSHEY_PLAIN,
                                   2, (0, 255, 0), 2)
        return img, bboxs
