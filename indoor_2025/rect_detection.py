import cv2
import numpy as np
import time

class RectDetection:
    # ... (O __init__ permanece como está) ...
    def __init__(self, image, image_depth, color):
        self.image = image
        self.image_depth = image_depth
        self.color_name = color
        self.erode_iterations = 2
        self.dilate_iterations = 5
        self.color_ranges = {'green': ([38, 23, 0], [98, 118, 128])} 
        self.delta = 30 
        self.DEPTH_TOLERANCE_M = 0.05

    def detect_gate_and_get_error(self):
        """
        Detecta um portão (duas traves verticais verdes) usando filtro de profundidade e cor.
        """
        if self.image is None:
            return False, None, None, False

        # Variáveis de dimensão e centro da imagem
        img_h, img_w, _ = self.image.shape
        centro_x_img = img_w // 2
        centro_y_img = img_h // 2
        
        # Desenho das linhas de Delta (mantido da versão anterior)
        x_delta_esq = centro_x_img - self.delta
        x_delta_dir = centro_x_img + self.delta
        cv2.line(self.image, (centro_x_img, 0), (centro_x_img, img_h), (255, 255, 255), 1)
        cv2.line(self.image, (x_delta_esq, 0), (x_delta_esq, img_h), (0, 0, 255), 2)
        cv2.line(self.image, (x_delta_dir, 0), (x_delta_dir, img_h), (0, 0, 255), 2)

        if self.image_depth is None:
            print("Alerta: Imagem de profundidade (self.image_depth) não disponível.")
            return False, None, None, False

        image_depth = self.image_depth.copy()
        h, w = image_depth.shape
        crop_start = h // 3
        cropped_image = image_depth[crop_start:, :] 

        depth_valid = cropped_image[np.isfinite(cropped_image) & (cropped_image > 0)]
        
        if depth_valid.size == 0:
            print("Alerta: Nenhum dado de profundidade válido encontrado nos 2/3 inferiores.")
            return False, None, None, False

        min_depth = np.min(depth_valid)
        max_allowed_depth = min_depth + self.DEPTH_TOLERANCE_M
        
        depth_mask = (self.image_depth >= min_depth) & (self.image_depth <= max_allowed_depth)
        
        depth_mask_u8 = depth_mask.astype(np.uint8) * 255
        
        image = cv2.bitwise_and(self.image.copy(), self.image.copy(), mask=depth_mask_u8)
        
        cv2.putText(self.image, f"Min Z: {min_depth:.2f}m", (10, img_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(self.image, f"Max Z: {max_allowed_depth:.2f}m", (10, img_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        lower, upper = self.color_ranges.get(self.color_name, ([0, 0, 0], [0, 0, 0]))
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.erode(mask, None, iterations=self.erode_iterations)
        mask = cv2.dilate(mask, None, iterations=self.dilate_iterations)
        color_detected = cv2.countNonZero(mask) > 0

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        maior_esq, maior_dir = None, None
        area_max_esq, area_max_dir = 0, 0

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            
            # Filtro por proporção vertical (h > 3*w) e área mínima
            if h > 3*w and area > 250:
                cx = x + w // 2
                cy = y + h // 2
                
                if cx < centro_x_img - self.delta - 10: # Barra à Esquerda
                    if area > area_max_esq:
                        area_max_esq = area
                        maior_esq = (cx, cy, x, y, w, h)
                elif cx > centro_x_img + self.delta: # Barra à Direita
                    if area > area_max_dir:
                        area_max_dir = area
                        maior_dir = (cx, cy, x, y, w, h)

        gate_found = maior_esq is not None and maior_dir is not None
        if not gate_found:
            return False, None, None, color_detected

        # Desenha Bounding Boxes VERDES e o ponto de centro azul
        for rect_info in [maior_esq, maior_dir]:
            if rect_info is not None:
                _, _, x, y, w, h = rect_info
                # Desenha o retângulo verde sobre self.image
                cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calcula o ponto médio e desenha
        midpoint_x = (maior_esq[0] + maior_dir[0]) // 2
        midpoint_y = (maior_esq[1] + maior_dir[1]) // 2
        
        # Ponto de centro do portão (Azul)
        cv2.circle(self.image, (midpoint_x, midpoint_y), 7, (255, 0, 0), -1) 
        # Centro absoluto da imagem (Vermelho)
        cv2.circle(self.image, (centro_x_img, centro_y_img), 7, (0, 0, 255), -1)

        error_x = midpoint_x - centro_x_img
        error_y = centro_y_img - midpoint_y
        
        return True, error_x, error_y, color_detected



    def detect_gate_depth_and_get_error(self, depth_tolerance=0.2):
        """
        Detecta um portão com base em profundidade e contornos verdes significativos.
        Retorna:
            - gate_found (bool): True se houver objetos à esquerda e à direita a distâncias semelhantes
                                e se houver contornos verdes com área > 500 em ambos os lados.
            - error_x (float): diferença entre o centro da câmera e o ponto médio entre os objetos.
        """
        if self.image is None or self.image_depth is None:
            return False, None
        
        image = self.image.copy()
        image_depth = self.image_depth.copy()

        h, w = image_depth.shape
        crop_start = h // 3
        cropped_image = image_depth[crop_start:, :]


        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img_h, img_w, _ = image.shape
        centro_x_img = img_w // 2

        lower, upper = self.color_ranges.get(self.color_name, ([0, 0, 0], [0, 0, 0]))
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.erode(mask, None, iterations=self.erode_iterations)
        mask = cv2.dilate(mask, None, iterations=self.dilate_iterations)

        left_region = cropped_image[:, :centro_x_img - self.delta - 40]
        right_region = cropped_image[:, centro_x_img + self.delta:]

        left_valid = left_region[np.isfinite(left_region)]
        left_valid = left_valid[left_valid > 0]
        right_valid = right_region[np.isfinite(right_region)]
        right_valid = right_valid[right_valid > 0]

        if left_valid.size == 0 or right_valid.size == 0:
            return False, None

        left_min_depth = float(np.min(left_valid))
        right_min_depth = float(np.min(right_valid))

        if abs(left_min_depth - right_min_depth) > depth_tolerance:
            return False, None

        left_mask = mask[:, :centro_x_img - self.delta - 40]
        right_mask = mask[:, centro_x_img + self.delta:]

        contours_left, _ = cv2.findContours(left_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        large_contour_left = any(cv2.contourArea(cnt) > 500 for cnt in contours_left)

        contours_right, _ = cv2.findContours(right_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        large_contour_right = any(cv2.contourArea(cnt) > 500 for cnt in contours_right)

        if not (large_contour_left and large_contour_right):
            return False, None

        left_min_indices = np.where(left_region == left_min_depth)
        right_min_indices = np.where(right_region == right_min_depth)

        left_center_x = int(np.mean(left_min_indices[1])) if left_min_indices[1].size > 0 else 0
        right_center_x = int(np.mean(right_min_indices[1])) if right_min_indices[1].size > 0 else (img_w // 2)

        right_center_x += centro_x_img + self.delta + 40

        midpoint_x = (left_center_x + right_center_x) / 2
        error_x = midpoint_x - centro_x_img

        cv2.circle(image, (int(left_center_x), img_h // 2), 6, (0, 255, 255), -1)
        cv2.circle(image, (int(right_center_x), img_h // 2), 6, (0, 255, 255), -1)
        cv2.line(image, (int(left_center_x), img_h // 2), (int(right_center_x), img_h // 2), (255, 0, 0), 2)
        cv2.circle(image, (int(midpoint_x), img_h // 2), 6, (0, 0, 255), -1)

        return True, float(error_x)

    # Adicione ou substitua este método na sua classe RectDetection

    def is_at_target_distance(self, target_distance):
        """
        Verifica se o objeto mais próximo em uma imagem de profundidade JÁ CORTADA 
        está na distância alvo.

        Args:
            cropped_image_depth (np.array): A imagem de profundidade já cortada.
            target_distance (float): A distância desejada em metros.

        Returns:
            tuple[bool, float | None]:
                - bool: True se a distância mínima for menor ou igual à distância alvo.
                - float | None: A distância mínima encontrada, ou None se não houver dados válidos.
        """
        if self.image_depth is None:
            return False, None

        image_depth = self.image_depth.copy()

        h, w = image_depth.shape
        crop_start = h // 3
        cropped_image = image_depth[crop_start:, :]

        depth_valid = cropped_image[np.isfinite(cropped_image)]
        depth_valid = depth_valid[depth_valid > 0]

        if depth_valid.size == 0:
            return False, None  

        min_depth = float(np.min(depth_valid))

        reached_target = min_depth <= target_distance

        return reached_target, min_depth
    


