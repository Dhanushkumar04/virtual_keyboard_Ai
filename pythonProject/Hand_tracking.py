import cv2
import mediapipe as mp
import time
import math  # Import math for distance calculation


class HandDetector:
    def __init__(self, staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = float(detectionCon)  # Ensure this is a float
        self.minTrackCon = float(minTrackCon)  # Ensure this is a float
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.staticMode,
                                        max_num_hands=self.maxHands,
                                        model_complexity=self.modelComplexity,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)

        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handIndex=0, draw=True):
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handIndex]
            h, w, _ = img.shape
            lmList = []
            for id, lm in enumerate(hand.landmark):
                px, py = int(lm.x * w), int(lm.y * h)
                lmList.append([px, py])  # Store only 2D positions

            xList = [lm[0] for lm in lmList]
            yList = [lm[1] for lm in lmList]
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = (xmin, ymin, xmax - xmin, ymax - ymin)

            if draw:
                self.mpDraw.draw_landmarks(img, hand, self.mpHands.HAND_CONNECTIONS)
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                              (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                              (255, 0, 255), 2)

            return lmList, bbox

        return [], []  # Always return empty lists if no hands are detected

    def findDistance(self, index1, index2, img, draw=False):
        """Calculate the distance between two landmarks by their indices."""
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]  # Assume we're using the first hand
            h, w, _ = img.shape

            # Get the landmark positions
            lm1 = hand.landmark[index1]
            lm2 = hand.landmark[index2]

            # Convert normalized coordinates to pixel values
            x1, y1 = int(lm1.x * w), int(lm1.y * h)
            x2, y2 = int(lm2.x * w), int(lm2.y * h)

            # Calculate the distance
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if draw:
                # Draw circles on the landmarks
                cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)  # Circle for first point
                cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)  # Circle for second point
                # Draw a line between the points
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)

            return distance

        return 0  # Return 0 if no hands are detected


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)

        # Calculate the distance between index (8) and middle (12) fingers
        distance = detector.findDistance(8, 12, img, draw=True)
        if distance > 0:
            print(f"Distance between index and middle finger: {distance:.2f}")

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
