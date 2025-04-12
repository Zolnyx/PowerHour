print("✅ Starting app_squat.py...")

import dash
from dash import dcc, html
# import dash_dangerously_set_inner_html
import mediapipe as mp
import SquatPosture as sp
from flask import Flask, Response
import cv2
import numpy as np
import torch  # Changed from tensorflow to torch
from utils import landmarks_list_to_array, label_params, label_final_results

print("✅ Imported all libraries successfully.")

# Initialize MediaPipe
print("🔹 Loading MediaPipe modules...")
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
print("✅ Loaded MediaPipe modules.")

# Load PyTorch model
print("🔹 Loading PyTorch model...")
class SquatPostureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(5, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 6)  # 6 outputs for [c, k, h, r, x, i]
        )
        
    def forward(self, x):
        return self.net(x)

model = SquatPostureModel()
model.load_state_dict(torch.load("squat_model.pth"))
model.eval()
print("✅ PyTorch model loaded successfully.")

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
    with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            image = cv2.flip(image, 1)

            if not success:
                print("Ignoring empty camera frame.")
                break

            # Process frame with MediaPipe
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Get squat parameters
            params = sp.get_params(results)
            if params is None:
                continue

            # Convert to PyTorch tensor and predict
            with torch.no_grad():
                input_tensor = torch.FloatTensor(params.reshape(1, -1))
                output = torch.sigmoid(model(input_tensor)).numpy()[0]  # Apply sigmoid for probabilities

            # Process output (same logic as before)
            output_name = ['c', 'k', 'h', 'r', 'x', 'i']
            label = "".join([output_name[i] for i in range(6) if output[i] > 0.5])
            label = "c" if label == "" else label  # Default to correct if no flags
            
            # Add visual feedback
            label_final_results(image, label)

            # Encode frame for web streaming
            ret, jpeg = cv2.imencode('.jpg', image)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Flask/Dash setup
server = Flask(__name__)
app = dash.Dash(__name__, server=server)
app.title = "Posture"

@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

app.layout = html.Div(className="main", children=[
    html.Link(rel="stylesheet", href="/assets/stylesheet.css"),
    html.Div(className="main-container", children=[
        html.Table(className="table", children=[
            html.Tr(className="row", children=[
                html.Td(html.Img(src="/assets/animation_for_web.gif", className="logo"))
            ]),
            html.Tr(className="choices", children=[
                html.Td("Your personal AI Gym Trainer")
            ]),
            html.Tr(className="row", children=[
                html.Td(html.Img(src="/video_feed", className="feed"))
            ]),
            html.Tr(className="disclaimer", children=[
                html.Td("Please ensure that the scene is well lit and your entire body is visible")
            ])
        ])
    ])
])

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run_server(debug=True)