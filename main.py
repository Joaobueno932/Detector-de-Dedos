import cv2
import mediapipe as mp

# Inicializando captura de vídeo e MediaPipe
video = cv2.VideoCapture(0)
hands = mp.solutions.hands
Hands = hands.Hands(max_num_hands=2)  # Permitindo até 2 mãos
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = video.read()
    if not success:
        print("Falha ao capturar imagem")
        break

    frameRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Hands.process(frameRGB)
    handPoints = results.multi_hand_landmarks
    h, w, _ = img.shape
    total_contador = 0

    if handPoints:
        for i, handLms in enumerate(handPoints):
            mpDraw.draw_landmarks(img, handLms, hands.HAND_CONNECTIONS)
            pontos = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                pontos.append((cx, cy))

            # Verifica se é mão esquerda ou direita
            hand_label = results.multi_handedness[i].classification[0].label

            dedos = [8, 12, 16, 20]
            contador = 0
            if pontos:
                if hand_label == "Right":
                    if pontos[4][0] < pontos[3][0]:
                        contador += 1
                else:  # Left hand
                    if pontos[4][0] > pontos[3][0]:
                        contador += 1

                for x in dedos:
                    if pontos[x][1] < pontos[x-2][1]:
                        contador += 1

            total_contador += contador

    cv2.rectangle(img, (80, 10), (200, 110), (255, 0, 0), -1)
    cv2.putText(img, str(total_contador), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5)

    cv2.imshow('Imagem', img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty('Imagem', cv2.WND_PROP_VISIBLE) < 1:
        break

video.release()
cv2.destroyAllWindows()
