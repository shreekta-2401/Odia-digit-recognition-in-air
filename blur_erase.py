import cv2
import mediapipe as mp
import numpy as np
import time
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Function to extract hand landmarks
def extract_hand_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    landmarks = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                landmarks.append((landmark.x, landmark.y))
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return landmarks, image

# Function to check for open fist gesture (all fingers extended)
def is_open_fist(landmarks):
    # Check if all fingers are extended (y-coordinates of finger tips are above y-coordinates of corresponding lower joints)
    if not landmarks:
        return False
    return all(
        landmarks[i][1] < landmarks[i - 2][1]
        for i in [8, 12, 16, 20]  # Index, middle, ring, and pinky finger tips
    )

# Function to check for thumbs up gesture
def is_thumbs_up(landmarks):
    # Check if thumb is up and other fingers are down
    if not landmarks:
        return False
    thumb_up = landmarks[4][1] < landmarks[3][1]
    other_fingers_down = all(
        landmarks[i][1] > landmarks[i - 2][1]
        for i in [8, 12, 16, 20]  # Index, middle, ring, and pinky finger tips
    )
    return thumb_up and other_fingers_down

# Define the region around the fingertip to unblur
def unblur_region(mask, x, y, size=10):
    cv2.circle(mask, (x, y), size, (255), thickness=-1)
    return mask

def process_webcam_feed(output_dir='output'):
    cap = cv2.VideoCapture(0)
    clear_mask = None
    unblur_active = False
    open_fist_start_time = None
    last_fingertip_position = None

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Flip the frame to unmirror the image
        frame = cv2.flip(frame, 1)

        if clear_mask is None:
            clear_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        blurred_frame = cv2.GaussianBlur(frame, (51, 51), 0)
        
        # Create a lighter green overlay
        green_tint = np.zeros_like(frame)
        green_tint[:, :, 1] = 100  # Set green channel to a lighter value

        # Blend the blurred frame with the green overlay
        blended_blur = cv2.addWeighted(blurred_frame, 0.7, green_tint, 0.3, 0)

        landmarks, _ = extract_hand_landmarks(frame)

        if landmarks:
            fingertip_x = int(landmarks[8][0] * frame.shape[1])  # Index finger tip x
            fingertip_y = int(landmarks[8][1] * frame.shape[0])  # Index finger tip y

            if is_open_fist(landmarks):
                if open_fist_start_time is None:
                    open_fist_start_time = time.time()
                elif time.time() - open_fist_start_time >= 1:
                    unblur_active = True
            else:
                open_fist_start_time = None

            if is_thumbs_up(landmarks):
                unblur_active = False

            if last_fingertip_position is not None and unblur_active:
                if (fingertip_x, fingertip_y) == last_fingertip_position:
                    unblur_active = False

            last_fingertip_position = (fingertip_x, fingertip_y)

            if unblur_active:
                clear_mask = unblur_region(clear_mask, fingertip_x, fingertip_y)

        clear_mask_3ch = cv2.merge([clear_mask, clear_mask, clear_mask])
        clear_area = cv2.bitwise_and(frame, clear_mask_3ch)
        blurred_area = cv2.bitwise_and(blended_blur, cv2.bitwise_not(clear_mask_3ch))
        output_image = cv2.add(clear_area, blurred_area)

        # Create the monochrome effect (black background, white unblur)
        monochrome_image = np.zeros_like(frame)
        monochrome_image[clear_mask > 0] = 255

        cv2.imshow('Blur Erase', output_image)

        if not unblur_active and frame_count > 0:
            # Save the monochrome image
            gray_frame = cv2.cvtColor(monochrome_image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(output_dir, f'frame_{frame_count}.png'), gray_frame)
            frame_count = 0  # Reset frame count after saving
        elif unblur_active:
            frame_count += 1  # Increment frame count while unblur is active

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_webcam_feed()
