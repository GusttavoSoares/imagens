import cv2

# testando camera trocar o 0
capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Erro para abrir a câmera")
    exit()

while True:
    ret, frame = capture.read()
    if not ret:
        print("erro p ler frame")
        break

    # Pega as dimensões originais
    (h, w) = frame.shape[:2]

    # Define nova largura e calcula altura 
    new_width = 640
    new_height = int((new_width / w) * h)

    # Redimensiona
    resized_image = cv2.resize(frame, (new_width, new_height))

    # Mostra a imagem
    cv2.imshow("cap", resized_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
