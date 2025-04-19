print("✅ Starting app_squat.py...")  # Debug print

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_dangerously_set_inner_html
import mediapipe as mp
import SquatPosture as sp
from flask import Flask, Response
import cv2
import tensorflow as tf
import numpy as np
import time
from utils import landmarks_list_to_array, label_params, label_final_results

print("✅ Imported all libraries successfully.")  # Debug print

print("🔹 Loading MediaPipe modules...")
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
print("✅ Loaded MediaPipe modules.")

print("🔹 Loading model...")
model = tf.keras.models.load_model("working_model_1")
print("✅ Model loaded successfully.")

class VideoCamera(object):
    def __init__(self):
        print("🎥 Trying to access webcam...")
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            print("❌ ERROR: Cannot access webcam. Exiting.")
            exit()
        print("✅ Webcam accessed successfully.")

    def __del__(self):
        self.video.release()

def gen(camera):
    cap = camera.video
    i = 0
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            image = cv2.flip(image, 1)
            image_height, image_width, _ = image.shape

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            dim = (image_width // 5, image_height // 5)
            resized_image = cv2.resize(image_rgb, dim)

            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            results = pose.process(resized_image)
            params = np.ravel(sp.get_params(results))

            if len(params) < 5:
                continue  # Skip invalid frame

            flat_params = params.reshape(1, 5)
            output = model.predict(flat_params)

            # Custom weighting
            output[0][0] *= 0.7
            output[0][1] *= 1.7
            output[0][2] *= 4
            output[0][3] *= 0
            output[0][4] *= 5
            output = output * (1 / np.sum(output))
            output[0][2] += 0.1

            output_name = ['c', 'k', 'h', 'r', 'x', 'i']
            label = "".join([output_name[i] for i in range(1, 4) if output[0][i] > 0.5])
            label = "c" if label == "" else label
            label += 'x' if output[0][4] > 0.15 and label == 'c' else ''

            label_final_results(image, label)
            i += 1

            ret, jpeg = cv2.imencode('.jpg', image)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Flask + Dash setup
server = Flask(__name__)
app = dash.Dash(__name__, server=server)
app.title = "Power-Hour"

@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

# Dash layout with GIF → logo swap JS
app.layout = html.Div(className="main", children=[
    html.Link(
        rel="stylesheet",
        href="/assets/styles.css"
    ),
    dash_dangerously_set_inner_html.DangerouslySetInnerHTML("""
        <div class="main-container">
            <table cellspacing="20px" class="table">
                <tr class="row">
                    <td>
                        <img id="splash" src="/assets/powerhour_animation.gif" class="logo" />
                    </td>
                </tr>
                <tr class="choices">
                    <td> Your personal AI Gym Trainer for Squats </td>
                </tr>
                <tr class="row">
                    <td> <img src="/video_feed" class="feed"/> </td>
                </tr>
                <tr class="disclaimer">
                    <td> Please ensure that the scene is well lit and your entire body is visible </td>
                </tr>
            </table>
        </div>
        <script>
            const splash = document.getElementById('splash');
            const logoReplacement = "/assets/powerhour_logo.png";
            setTimeout(() => {
                splash.src = logoReplacement;
            }, 2500);  // Match your GIF duration
        </script>
    """)
])

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    app.run_server(debug=True,use_reloader=False)
