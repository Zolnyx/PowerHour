# Posture: Pose Tracking and Machine Learning for prescribing corrective suggestions to improve posture and form while exercising.

This repository contains code made for submission to [Atlas Hacks.](https://devpost.com/software/posture-w5670m)

Our project is an AI-based Personalised Exercise Feedback Assistant: an algorithm that views your exercise posture in real time and tells what you're getting right, and what you're getting wrong! 


https://github.com/Amitesh218/PowerHour/assets/15158326/b5155d6b-667e-4389-b75b-9d3257b7202c

# Our demo

To run the app, first install python 3.9, then:

initialise a virtual environment using python 
```zsh
py -3.9 -m venv myenv
source myenv/bin/activate # for linux
.\myenv\Scripts\Activate # for windows
```
following that, install the requirements
```zsh
pip install -r requirements.txt
```
after that simply run
```zsh
python -u app_squat.py
```

# Our model

With no available dataset online, we took it upon ourselves to generate data. After collecting hours of labelled videos of people performing Squats in a multitude of correct and incorrect ways, we used each frame of video (at 12fps) as a labeled training example - which got us a training set size of tens of thousands. 
