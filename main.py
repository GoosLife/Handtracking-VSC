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
mpHands = mp.solutions.hands
hands = mpHands.Hands()

# Face detection
mpFaceMesh = mp.solutions.face_mesh

# Gesture model path
model_directory = os.getcwd()
model_file ="gesture_recognizer.task"

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
    base_options = BaseOptions(model_asset_path=os.path.join(model_directory, model_file)),
    running_mode = VisionRunningMode.LIVE_STREAM,
    result_callback = print_gesture) 
with GestureRecognizer.create_from_options(options) as recognizer, mpFaceMesh.FaceMesh(
    max_num_faces = 2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(imageRGB)
        faceResults = face_mesh.process(image)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.GetMat(image))

        recognizer.recognize_async(mp_image, cap.get(cv2.CAP_PROP_POS_MSEC))

        gesture = "NONE"

        
        # checking whether a hand is detected
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks: # working with each hand
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    if id == 10:
                        cv2.circle(image, (cx, cy), 12, (255,0,0), cv2.FILLED)
                        gesture = "LANDMARK ID 10 DETECTED"

                mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS) 
        

        # checking whether a face is detected
        if faceResults.multi_face_landmarks:
            for face_landmarks in faceResults.multi_face_landmarks:
                mpDraw.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mpFaceMesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mpDraw.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mpFaceMesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mpDraw.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mpFaceMesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
                
        image = cv2.flip(image, 1)
        # write gesture to img
        cv2.putText(image, str(gesture), (25,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 5)
                
        cv2.imshow("Output", image)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()