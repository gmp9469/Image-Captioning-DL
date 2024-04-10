from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("RPSmodel.h5")
categories = {0: "rock", 1: "paper", 2: "scissors"}

def generate_frames():
    vid = cv2.VideoCapture(0)
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        roi = frame[100:400, 100:400]
        img = cv2.resize(roi, (300, 300))
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        move = categories[np.argmax(prediction)]
        cv2.putText(frame, move, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    vid.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
