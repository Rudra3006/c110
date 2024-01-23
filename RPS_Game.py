import cv2
import mediapipe as mp
import random
from collections import deque
import statistics as st


def compute_fingers(hand_landmarks, count):

    if hand_landmarks[8][2] < hand_landmarks[6][2]:
        count += 1

    if hand_landmarks[12][2] < hand_landmarks[10][2]:
        count += 1

    if hand_landmarks[16][2] < hand_landmarks[14][2]:
        count += 1

    if hand_landmarks[20][2] < hand_landmarks[18][2]:
        count += 1

    if hand_landmarks[4][3] == "Left" and hand_landmarks[4][1] > hand_landmarks[3][1]:
        count += 1
    elif hand_landmarks[4][3] == "Right" and hand_landmarks[4][1] < hand_landmarks[3][1]:
        count += 1
    return count

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

webcam = cv2.VideoCapture(0)

cpu_choices = ["Rock", "Paper", "Scissors"]
cpu_choice = "Nothing"
cpu_score, player_score = 0, 0
winner_colour = (0, 255, 0)
player_choice = "Nothing"
hand_valid = False
display_values = ["Rock", "Invalid", "Scissors", "Invalid", "Invalid", "Paper"]
de = deque(['Nothing'] * 5, maxlen=5)

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    while webcam.isOpened():
        success, image = webcam.read()
        if not success:
            print("Camera isn't working")
            continue
        

        image = cv2.flip(image, 1)

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        handNumber = 0
        hand_landmarks = []
        isCounting = False
        count = 0

        if results.multi_hand_landmarks:
            isCounting = True

            for hand in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                label = results.multi_handedness[handNumber].classification[0].label

                for id, landmark in enumerate(hand.landmark):
                    imgH, imgW, imgC = image.shape
                    xPos, yPos = int(landmark.x *
                                     imgW), int(landmark.y * imgH)

                    hand_landmarks.append([id, xPos, yPos, label])

                count = compute_fingers(hand_landmarks, count)

                handNumber += 1
        else:
            hand_valid = False

        if isCounting and count <= 5:
            player_choice = display_values[count]
        elif isCounting:
            player_choice = "Invalid"
        else:
            player_choice = "Nothing"

        de.appendleft(player_choice)

        try:
            player_choice = st.mode(de)
        except st.StatisticsError:
            print("Stats Error")
            continue

        cv2.putText(image, "You", (90, 75),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 5)

        cv2.putText(image, "CPU", (1050, 75),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 5)

        cv2.putText(image, player_choice, (45, 375),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 5)

        cv2.putText(image, cpu_choice, (1000, 375),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 5)

        cv2.putText(image, winner, (530, 650),
                    cv2.FONT_HERSHEY_DUPLEX, 2, winner_colour, 5)

        cv2.putText(image, str(player_score), (145, 200),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 5)

        cv2.putText(image, str(cpu_score), (1100, 200),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 5)

        cv2.imshow('Rock, Paper, Scissors', image)

        if cv2.waitKey(1) & 0xFF == 27:
            break

webcam.release()
cv2.destroyAllWindows()
