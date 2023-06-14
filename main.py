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
cap = cv2.VideoCapture(0) # REFACTORED

# Drawing utilities
mpDraw = mp.solutions.drawing_utils # REFACTORED
mp_drawing_styles = mp.solutions.drawing_styles #REFACTORED

# Face detection
mpFaceMesh = mp.solutions.face_mesh # OMITTED

# Hand detection
mpHands = mp.solutions.hands
hands = mpHands.Hands()

while cap.isOpened():
    success, image = cap.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)
    
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

        if (finger_utils.pointing_left(rightHand)):
            rightGesture = "POINT [L]"
        elif (finger_utils.pointing_right(rightHand)):
            rightGesture = "POINT [R]"
        elif (finger_utils.hand_down(rightHand)):
            rightGesture = "HAND DOWN"

        if (finger_utils.pointing_left(leftHand)):
            leftGesture = "POINT [L]"
        elif (finger_utils.pointing_right(leftHand)):
            leftGesture = "POINT [R]"
        elif (finger_utils.hand_down(leftHand)):
            leftGesture = "HAND DOWN"
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
    cv2.putText(image, str(rightGesture), (225,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 5)
            
    cv2.imshow("Output", image)

    pressedKey = cv2.waitKey(1) & 0xFF

    if pressedKey != 255:
        if pressedKey == 27:
            break
        elif pressedKey == ord('\b'):
            print("\b", end=" \b")
        elif pressedKey == 13: # Enter key
            print('\n', end='\r')
        else:
            print(chr(pressedKey), end="")