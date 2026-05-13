import numpy as np
import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Webcam
cap = cv2.VideoCapture(0)

# Hand detector
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Screen size
screen_width, screen_height = pyautogui.size()

# Previous mouse location
prev_x = 0
prev_y = 0

# Smoothing factor
smoothening = 5

# Click cooldown
last_click_time = 0

while True:

    success, frame = cap.read()

    # Flip frame
    frame = cv2.flip(frame, 1)

    frame_height, frame_width, _ = frame.shape

    # RGB conversion
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Hand detection
    result = hands.process(rgb_frame)

    # Draw control area
    cv2.rectangle(
        frame,
        (100, 100),
        (540, 380),
        (255, 0, 255),
        2
    )

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Index finger tip
            index = hand_landmarks.landmark[8]

            # Thumb tip
            thumb = hand_landmarks.landmark[4]

            # Convert coordinates
            x = int(index.x * frame_width)
            y = int(index.y * frame_height)

            # Draw finger point
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

            # Convert webcam area to screen size
            screen_x = np.interp(
                x,
                (100, 540),
                (0, screen_width)
            )

            screen_y = np.interp(
                y,
                (100, 380),
                (0, screen_height)
            )

            # Smooth movement
            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening

            pyautogui.moveTo(curr_x, curr_y)

            prev_x = curr_x
            prev_y = curr_y

            # Distance between thumb and index
            thumb_x = int(thumb.x * frame_width)
            thumb_y = int(thumb.y * frame_height)

            distance = math.hypot(
                thumb_x - x,
                thumb_y - y
            )

            # Click gesture
            if distance < 30:

                current_time = time.time()

                # Prevent rapid clicking
                if current_time - last_click_time > 0.5:

                    pyautogui.click()

                    last_click_time = current_time

                    cv2.circle(
                        frame,
                        (x, y),
                        20,
                        (0, 0, 255),
                        -1
                    )

    cv2.imshow("Advanced AI Virtual Mouse", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()