from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import base64
import io
from PIL import Image

app = Flask(__name__)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

@app.route('/')
def index():
    return render_template('zxc.html')

@app.route('/detect_hand', methods=['POST'])
def detect_hand():
    data = request.json
    img_data = data['image']

    # Відрізаємо префікс base64
    img_str = img_data.split(',')[1]

    # Декодуємо base64 в numpy array
    img_bytes = base64.b64decode(img_str)
    image = Image.open(io.BytesIO(img_bytes))
    image = image.convert('RGB')
    frame = np.array(image)

    # Обробка Mediapipe (BGR формат)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = hands.process(frame)

    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                landmarks.append({'id': id, 'x': lm.x, 'y': lm.y, 'z': lm.z})

    return jsonify({'landmarks': landmarks})

if __name__ == '__main__':
    app.run(debug=True)
