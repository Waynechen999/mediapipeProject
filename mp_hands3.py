import cv2
import mediapipe as mp
import numpy as np
import pygame
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize Pygame for sound
pygame.mixer.init()
hit_sound = pygame.mixer.Sound("C:\\testAI\\mediapipeProject\\sound.mp3")
minus_sound = pygame.mixer.Sound("C:\\testAI\\mediapipeProject\\minus.mp3")

def circle(point, circle_center, circle_radius):
    return np.linalg.norm(np.array(point) - np.array(circle_center)) <= circle_radius

# For static images:
IMAGE_FILES = []
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    # Create a list to store falling balls
    circles = []

    for idx, file in enumerate(IMAGE_FILES):
        # Read an image, flip it around y-axis for correct handedness output.
        image = cv2.imread(file)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            image_height, image_width, _ = image.shape

        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                hand_coord = (int(landmark.x * image_width),
                              int(landmark.y * image_height))

                for x in circles:
                        if circle(hand_coord, x['center'], x['radius']):
                            x['click_count'] += 1
                            if (x['click_count'] == 1) and (x['color'] == (0, 255, 0)):
                                score += 1
                                print("+1 point")
                            elif (x['click_count'] == 1) and (x['color'] == (0, 0, 255)):
                                score -= 1
                                print("-1 point")
                                minus_sound.play()
                            x['color'] = (255,0,0)
                            
                            if x['click_count'] >= 3:
                                circles.remove(x)
                            hit_sound.play()

                for x in circles:
                    x['center'] = (x['center'][0], x['center'][1] + x['speed'])

                # Add a new ball with a certain probability
        if np.random.rand() < 0.05:  # 掉球的頻率
            color = (0, 255, 0) if np.random.rand() < 0.8 else (0, 0, 255)  # 80% chance for green, 20% for red
            new_ball = {
                'center': (np.random.randint(50, 500), 0),
                'radius': 30,
                'color': color,
                'opacity': 1.0,
                'click_count': 0,
                'alpha': 255,
                'speed': np.random.uniform(0.01, 1)
            }
            circles.append(new_ball)

            # Draw the circle on the image
        overlay = image.copy()
        for x in circles:
            cv2.circle(overlay, (int(x['center'][0]), int(x['center'][1])), x['radius'], x['color'], -1) # -1 fills the circle
            x['opacity'] = max(0, min(1, x['alpha'] / 255.0))
        cv2.addWeighted(overlay, 1.0, image, 0.5, 0, image)

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw pose landmarks on the image.
        mp_drawing.draw_landmarks(
            annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

        # Plot pose world landmarks
        mp_drawing.plot_landmarks(
            results.multi_hand_landmarks, mp_hands.HAND_CONNECTIONS)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    circles = []
    score = 0
    start_time = time.time()
    countdown_duration = 30
    timer_font = cv2.FONT_HERSHEY_SIMPLEX
    timer_font_size = 1
    timer_font_color = (0, 0, 0)
    timer_font_thickness = 2

    hand_landmarks = None  # Initialize hand_landmarks outside the loop

    pTime = 0  # Initialize pTime before the loop starts

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        # 1. 翻轉 & 轉 RGB
        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. 設唯讀，進行 Mediapipe 推論
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # 3. 推論完畢，設回可寫
        image.flags.writeable = True
        # 4. 轉回 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # =====  下面才做所有需要「寫」影像的操作  =====
    
        # a. 取得計時文字、分數文字
        remaining_time = max(countdown_duration - int(time.time() - start_time), 0)
        timer_text = f'Time: {remaining_time} s'

        # b. OpenCV 繪圖
        cv2.putText(image, timer_text, (10, 70), timer_font, timer_font_size, timer_font_color, timer_font_thickness)
       

        # Check if hands are detected before iterating
        if results.multi_hand_landmarks:
            image_height, image_width, _ = image.shape  # Move this line inside the loop
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    hand_coord = (int(landmark.x * image_width),
                                  int(landmark.y * image_height))

                    for x in circles:
                        if circle(hand_coord, x['center'], x['radius']):
                            x['click_count'] += 1
                            if (x['click_count'] == 1) and (x['color'] == (0, 255, 0)):
                                score += 1
                                print("+1 point")
                            elif (x['click_count'] == 1) and (x['color'] == (0, 0, 255)):
                                score -= 1
                                print("-1 point")
                                minus_sound.play()
                            x['color'] = (255,0,0)
                            
                            if x['click_count'] >= 3:
                                circles.remove(x)
                            hit_sound.play()
                                
                            
                    for x in circles:
                        x['center'] = (x['center'][0], x['center'][1] + x['speed'])

        # Add a new ball with a certain probability
        if np.random.rand() < 0.05:  # 掉球的頻率
            color = (0, 255, 0) if np.random.rand() < 0.8 else (0, 0, 255)  # 80% chance for green, 20% for red
            new_ball = {
                'center': (np.random.randint(50, 500), 0),
                'radius': 30,
                'color': color,
                'opacity': 1.0,
                'click_count': 0,
                'alpha': 255,
                'speed': np.random.uniform(0.01, 1)
            }
            circles.append(new_ball)

        overlay = image.copy()
        for x in circles:
            cv2.circle(overlay, (int(x['center'][0]), int(x['center'][1])), x['radius'], x['color'], -1)
            x['opacity'] = max(0, min(1, x['alpha'] / 255.0))
        cv2.addWeighted(overlay, 1.0, image, 0.5, 0, image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.putText(image, f'Score: {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(image, f'FPS: {int(fps)}', (10, 110), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)
            
        cv2.imshow('MediaPipe Hands', image)

        if time.time() - start_time > countdown_duration:
            print(f'Game Over! Total Score: {score}')
            user_input = input("Play again? (y/n): ")
            if user_input.lower() == 'y':
                # Reset the game
                circles = []
                score = 0
                start_time = time.time()
            else:
                break

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
