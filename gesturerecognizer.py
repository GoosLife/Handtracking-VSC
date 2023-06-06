import finger_utils

import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Video input 
cap = cv2.VideoCapture(0)

# Drawing utilities
mpDraw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Hand detection
#mpHands = mp.solutions.hands
hands = mpHands.Hands()

# Face detection
#mpFaceMesh = mp.solutions.face_mesh

# Gesture detection
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def print_gesture(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    print('gesture recognition result: {}'.format(result))

# Create a gesture recognizer instance with video mode
options = GestureRecognizerOptions(
    base_options = BaseOptions(model_asset_path="gesture_recognizer.task"),
    running_mode = VisionRunningMode.LIVE_STREAM,
    result_callback = print_gesture) 
with GestureRecognizer.create_from_options(options) as recognizer:
    print("Hello, World!")