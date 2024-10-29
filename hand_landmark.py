import cv2
import mediapipe as mp

# Inisialisasi Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
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

    # Menampilkan hasil
    cv2.imshow('Landmark', image)

    if cv2.waitKey(5) & 0xFF == 27:  # Tekan 'Esc' untuk keluar
        break

cap.release()
cv2.destroyAllWindows()