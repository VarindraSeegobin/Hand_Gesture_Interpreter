import cv2
import mediapipe as mp

class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, hand, self.mpHands.HAND_CONNECTIONS)

        return frame

    def findPosition(self, frame, handnum=0, draw=True):
        ls = []
        if hasattr(self, 'results') and self.results.multi_hand_landmarks:
            currentHand = self.results.multi_hand_landmarks[handnum]

            for id, lm in enumerate(currentHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                ls.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return ls

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detector.findHands(frame,False)
        cv2.imshow("Hand Detection", frame)
        detector.findPosition(frame)
        llist = detector.findPosition(frame)
        if len(llist) > 0:
            print(llist[1])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
