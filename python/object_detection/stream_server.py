from flask import Flask, Response
import cv2
import time

app = Flask(__name__)
latest_frame = None

@app.route("/")
def index():
    return "Go to /video_feed to see camera stream."

@app.route("/video_feed")
def video_feed():
    def generate():
        global latest_frame
        while True:
            if latest_frame is not None:
                ret, buffer = cv2.imencode('.jpg', latest_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.03)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
