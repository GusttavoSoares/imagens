import rclpy
import numpy as np
import cv2
import time

from enum import Enum
from communication2 import Mav

ALTITUDE = 1.25

class Detection:
    def detect_H_with_circle(img, debug=False):
        # --- 1. Leitura e pré-processamento ---
        if img is None:
            return False, None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # --- 2. Binarização (H e círculo são pretos, fundo é branco) ---
        _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)

        # --- 3. Detectar contornos ---
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found_circle = False
        H_detected = False
        center = None

        for c in contours:
            area = cv2.contourArea(c)
            if area < 500:  # ignora ruído
                continue

            # --- 4. Tentar identificar o círculo ---
            (x, y), radius = cv2.minEnclosingCircle(c)
            circle_area = np.pi * radius * radius
            if abs(circle_area - area) / circle_area < 0.25:  # é quase circular
                found_circle = True
                center = (int(x), int(y))
                cv2.circle(img, center, int(radius), (0, 255, 0), 2)

                # --- 5. Recortar região dentro do círculo ---
                mask = np.zeros_like(thresh)
                cv2.circle(mask, center, int(radius*0.9), 255, -1)
                roi = cv2.bitwise_and(thresh, mask)

                # --- 6. Detectar retângulos internos ("H") ---
                contours_inside, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rects = [cv2.boundingRect(cnt) for cnt in contours_inside if cv2.contourArea(cnt) > 100]
                
                # Esperamos 3 barras pretas (2 verticais + 1 horizontal)
                if len(rects) == 3:
                    rects = sorted(rects, key=lambda r: r[0])
                    x0, y0, w0, h0 = rects[0]
                    x1, y1, w1, h1 = rects[1]
                    x2, y2, w2, h2 = rects[2]

                    # Heurística: duas verticais e uma horizontal central
                    heights = [h0, h1, h2]
                    widths = [w0, w1, w2]
                    if max(heights) > 3 * max(widths):  # verticais altas e finas
                        H_detected = True

                        for (x, y, w, h) in rects:
                            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        if debug:
            cv2.imshow("Threshold", thresh)
            cv2.imshow("Detected H and Circle", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if found_circle and H_detected:
            return True, center
        else:
            return False, None

class States(Enum):
    ALTITUDE = 0,
    SEARCH = 1,
    APPROACH = 2,
    STABILIZE = 3,
    ANALYZE = 4,
    LAND = 5

class Mission_4:
    def __init__ (self, mav: Mav, zed_node):
        self.mav = mav
        self.mav.get_logger().info("Mission 4 created")
        self.state = States.ALTITUDE
        self.zed_node = zed_node
        self.problem = False
        self.running = True
        camera_index = 0
        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            self.mav.get_logger().error(f"Não foi possível abrir a câmera no índice {camera_index}")
            self.state = States.ERROR

    def cleanup(self):
        """Libera o recurso da câmera ao final da missão."""
        self.mav.get_logger().info("Liberando a câmera.")
        self.camera.release()

    def run(self) -> bool:
        if self.state == States.ALTITUDE:
            self.mav.get_logger().info("[ALTITUDE] Estabilizndo altitude.")
            self.mav.goto(self.mav.pose.position.x, self.mav.pose.position.y, ALTITUDE)
            time.sleep(2.0)
            self.state = States.SEARCH
        
        if self.state == States.SEARCH:
            self.mav.get_logger().info("[ALTITUDE] Procurando linha da base de pouso.")
            start_time = time.time()
            timeout = 30  
            base_found = False
            while time.time() - start_time < timeout:
                ret, frame = self.camera.read()

                if not ret or frame is None:
                    self.mav.get_logger().warn("[SEARCH] Não foi possível ler o frame da câmera.")
                    time.sleep(0.1)
                    continue

                










    pass
