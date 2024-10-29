import cv2
import mediapipe as mp
import pyautogui
import math

# Inisialisasi Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)

mp_drawing = mp.solutions.drawing_utils

# Menggunakan webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mengubah warna dari BGR ke RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Mendeteksi tangan
    results = hands.process(image)

    # Menggambar landmark tangan pada frame
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Mengambil posisi landmark jari telunjuk dan jempol
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Menghitung jarak antara jari telunjuk dan jempol
            index_finger_x = index_finger_tip.x * frame.shape[1]
            index_finger_y = index_finger_tip.y * frame.shape[0]
            thumb_x = thumb_tip.x * frame.shape[1]
            thumb_y = thumb_tip.y * frame.shape[0]

            distance = math.sqrt((index_finger_x - thumb_x) ** 2 + (index_finger_y - thumb_y) ** 2)

            # Mengatur volume berdasarkan jarak
            if distance < 50:  # Jika jari telunjuk dan jempol berdekatan
                pyautogui.press('volumedown')
            elif distance > 100:  # Jika jari telunjuk dan jempol menjauh
                pyautogui.press('volumeup')

    # Menampilkan hasil
    cv2.imshow('Volume Control with Hand Gesture', image)

    if cv2.waitKey(5) & 0xFF == 27:  # Tekan 'Esc' untuk keluar
        break

cap.release()
cv2.destroyAllWindows()