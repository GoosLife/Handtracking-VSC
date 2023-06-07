import finger_utils

import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np

# Video input 
cap = cv2.VideoCapture(0)

# Drawing utilities
mpDraw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

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

# Audio device control setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volMin, volMax = volume.GetVolumeRange()[:2]

# Keep track of previous finger distance for zoom gestures
prevFingerDistance = 0

def print_gesture(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    print('gesture recognition result: {}'.format(result))

# Create a gesture recognizer instance with video mode
options = GestureRecognizerOptions(
    base_options = BaseOptions(model_asset_path=os.path.join(os.getcwd(), model_file)),
    running_mode = VisionRunningMode.LIVE_STREAM,
    result_callback = print_gesture) 
with GestureRecognizer.create_from_options(options) as recognizer, mpFaceMesh.FaceMesh(
    max_num_faces = 2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    # Hand detection
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()

    while cap.isOpened():
        success, image = cap.read()
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(imageRGB)
        faceResults = face_mesh.process(image)

        ##### TODO: Figure out how to implement gesture recognition
        # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.GetMat(image))

        # recognizer.recognize_async(mp_image, cap.get(cv2.CAP_PROP_POS_MSEC))

        # Multi hand gesture setup per frame

        RIGHT_HAND = None
        LEFT_HAND = None

        rightGesture = "NONE"
        leftGesture = "NONE"

        handLandmarks = []
        handIndexIterator = 0

        # checking whether a hand is detected
        if results.multi_hand_landmarks:
            for hand in results.multi_handedness:
                # If the label is Right, it is actually the left hand because the image is being flipped.
                if hand.classification[0].label == "Right":
                    LEFT_HAND = handIndexIterator
                else:
                    RIGHT_HAND = handIndexIterator
                handIndexIterator += 1
            for handLms in results.multi_hand_landmarks: # working with each hand
                hand = []
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    hand.append([id, cx, cy])

                    if id == 10:
                        cv2.circle(image, (cx, cy), 12, (255,0,0), cv2.FILLED)
                
                handLandmarks.append(hand)

                # Check if hand is down (for each hand)
                #if finger_utils.hand_down(handLms.landmark):
                #    gesture = "HAND DOWN"
                #elif finger_utils.palm_up(handLms.landmark):
                #    gesture = "PALM UP"
                
                mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
        
        # Get thumb and index coords (for each hand)
        if LEFT_HAND != None and RIGHT_HAND != None:
            rightHand = handLandmarks[RIGHT_HAND]
            leftHand = handLandmarks[LEFT_HAND]

            if (finger_utils.palm_up(leftHand)):
                leftGesture = "PALM UP"
                if (handLandmarks[RIGHT_HAND] != None):
                    # Get thumb and index coordinates
                    x1, y1 = rightHand[4][1], rightHand[4][2]
                    x2, y2 = rightHand[8][1], rightHand[8][2]
                    #Get size of hand
                    wristX, wristY = rightHand[0][1], rightHand[0][2]
                    ringX, ringY = rightHand[mpHands.HandLandmark.RING_FINGER_TIP][1], rightHand[mpHands.HandLandmark.RING_FINGER_TIP][2]

                # Draw line between thumb and index
                cv2.circle(image, (x1, y1), 12, (0,255,0), cv2.FILLED)
                cv2.circle(image, (x2, y2), 12, (0,255,0), cv2.FILLED)
                cv2.line(image, (x1, y1), (x2, y2), (0,255,0), 3)

                # Calculate volume based on distance between fingers
                length = hypot(x2 - x1, y2 - y1)
                handSize = hypot(ringX - wristX, ringY - wristY)
                if  handSize != 0:
                    length = (length / handSize) * 100
                print(length)
                vol = np.interp(length, [15, 220], [volMin, volMax])
            
                # Set volume
                volume.SetMasterVolumeLevel(vol, None)

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
        cv2.putText(image, str(leftGesture), (25,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 5)
                
        cv2.imshow("Output", image)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()