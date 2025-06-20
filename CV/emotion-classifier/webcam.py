from xml.etree.ElementTree import TreeBuilder
import cv2
import pickle
import numpy as np
from sklearn import model_selection
import tensorflow as tf 
import mediapipe as mp 

mp_face_detection = mp.solutions.face_detection

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
    
model = tf.keras.models.load_model('./models/cnn0.66-1.84.h5')

# web cam
cam = cv2.VideoCapture(1)

def detect_face(frame):
    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        H, W, _ = frame.shape
        
        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            x1, y1 = max(0, x1), max(0, y1)
            w, h = min(W - x1, w), min(H - y1, h)
            
            return x1, y1, x1 + w, y1 + h
        else: 
            return None, None, None, None
        
def crop_face(image, x1, y1, x2, y2):
    print(x1, y1, x2, y2)
    face = image[y1:y2, x1:x2, :].copy()
    face = cv2.resize(face, (122, 122))
    return face.astype(np.float32) / 255.0

while True:
    rec, image = cam.read()
    
    # detect face
    x1, y1, x2, y2 = detect_face(image)     
    
    # show bbox
    if x1:
        # model prediction 
        face = crop_face(image, x1, y1, x2, y2)
        face = np.expand_dims(face, axis=0)
        
        pred = model.predict(face)
        print(pred)

        y_off = 30
        bar_w = 200
        bar_h = 20
        
        for i in range(8):
            label = le.inverse_transform([i])[0]
            conf = pred[0][i]
            
            fill_w = int(bar_w * conf)
            cv2.rectangle(image, (10, y_off), (10 + bar_w, y_off + bar_h), (128, 128, 128), -1)
            color = (0, int(255*conf), int(255*(1-conf)))
            cv2.rectangle(image, (10, y_off), (10 + fill_w, y_off + bar_h), color, -1)
            text = f"{label}: {conf:.2%}"
            cv2.putText(image, text, (220, y_off + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            y_off += 30
                          

    cv2.imshow('eblan', image)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()