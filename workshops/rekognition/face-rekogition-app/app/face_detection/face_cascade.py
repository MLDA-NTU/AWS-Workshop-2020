import cv2
from app import flask_app


model_path = flask_app.config['HAAR_MODEL']
cascadeClassifier = cv2.CascadeClassifier()


def detect_face(frame, inHeight=300):
    frameHaar = frame.copy()

    height = frame.shape[0]
    width = frame.shape[1]
    
    inWidth = int((width / height) * inHeight)
    scaleHeight = height / inHeight
    scaleWidth = width / inWidth
    
    frameHaarResized = cv2.resize(frameHaar, (inWidth, inHeight))
    frameGray = cv2.cvtColor(frameHaarResized, cv2.COLOR_BGR2GRAY)
    
    faces = cascadeClassifier.detectMultiScale(frameGray)
    bboxes = []

    for (x1, y1, w, h) in faces:
        x2, y2 = x1 + w, y1 + h
        cvRect = [int(x1 * scaleWidth), int(y1 * scaleHeight),
                  int(x2 * scaleWidth), int(y2 * scaleHeight)]
        bboxes.append(cvRect)
        cv2.rectangle(frameHaar, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]),
                      (0, 255, 0), int(round(height / 150)), 4)
    
    return frameHaar, bboxes
