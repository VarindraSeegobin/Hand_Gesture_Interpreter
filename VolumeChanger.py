import math

import cv2
import numpy as np
import HandTracker as ht
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
minVol, maxVol = volume.GetVolumeRange()[:2]

print(minVol, maxVol)

cap = cv2.VideoCapture(0)

handtracker = ht.HandDetector(detectionCon=0.65, maxHands=1)

minVol, maxVol = volume.GetVolumeRange()[:2]

def set_volume_from_hand_distance(hand_distance, min_hand=0.3, max_hand=1.5):
    hand_distance = max(min(hand_distance, max_hand), min_hand)
    interpolated_dB = np.interp(hand_distance, [min_hand, max_hand], [minVol, maxVol])
    volume.SetMasterVolumeLevel(interpolated_dB, None)


while True:
    ret,frame = cap.read()
    a = handtracker.findHands(frame)
    landmarks = handtracker.findPosition(frame)
    b = handtracker.findPosition(frame)
    if len(landmarks) > 17:
        # Finger tips
        x1, y1 = landmarks[4][1], landmarks[4][2]  # Thumb tip
        x2, y2 = landmarks[8][1], landmarks[8][2]  # Index tip

        xr1, yr1 = landmarks[0][1], landmarks[0][2]  # Wrist
        xr2, yr2 = landmarks[9][1], landmarks[9][2]  # Middle finger base

        d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        ref = math.sqrt((xr2 - xr1) ** 2 + (yr2 - yr1) ** 2)
        norm_dist = round((d / ref if ref != 0 else 0),1)
        print(f"Normalized distance: {norm_dist:.2f}")
        set_volume_from_hand_distance(norm_dist)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.imshow('frame', a)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
