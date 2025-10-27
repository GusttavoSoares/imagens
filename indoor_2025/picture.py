from mission_base import ZedSubscriber
from rect_detection import RectDetection
import rclpy
import cv2
import time
import os
import datetime

class Picture:
    """
    Classe responsável por capturar um frame do tópico ROS da ZED, 
    executar a detecção do portão e salvar a foto original e a foto processada.
    """
    def __init__(self, zed_node: ZedSubscriber):
        self.zed_node = zed_node
        self.cv_image = None
        self.cv_depth_image = None # Mantido para compatibilidade com RectDetection
        self.output_dir = "captured_images"

        # Cria o diretório de saída se não existir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Diretório criado: {self.output_dir}")

    def update_frame(self):
        """Atualiza o frame da câmera através do spin do ROS 2."""
        rclpy.spin_once(self.zed_node, timeout_sec=0)
        self.cv_image = self.zed_node.frame
        self.cv_depth_image = self.zed_node.depth_frame
    
    def take_and_save_pictures(self, color='green'):
        """
        Espera por um frame, executa a detecção e salva as imagens.
        """
        print("Aguardando frame da ZED...")
        
        # Espera até receber um frame (máximo de 10 segundos)
        start_time = time.time()
        timeout = 10.0
        while self.cv_image is None and self.cv_depth_image is None and (time.time() - start_time) < timeout:
            self.update_frame()
            time.sleep(0.1)

        if self.cv_image is None:
            print("ERRO: Não foi possível receber o frame da câmera após 10 segundos.")
            return
        
        if self.cv_depth_image is None:
            print("ERRO: Não foi possível receber o frame da câmera após 10 segundos.")
            return

        # Gera o timestamp único para os nomes dos arquivos
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # --- 1. SALVAR FOTO ORIGINAL (Cópia Limpa) ---
        # Faz uma cópia limpa antes de passar para o detector (que desenha sobre a imagem)
        original_image = self.cv_image.copy()

        original_filename = os.path.join(self.output_dir, f"original_{timestamp}.png")
        cv2.imwrite(original_filename, original_image)
        print(f"1. Foto original salva: {original_filename}")


        # --- 2. DETECÇÃO E SALVAR FOTO PROCESSADA ---
        # O RectDetection desenhará os retângulos em self.cv_image
        
        # Nota: RectDetection precisa da imagem RGB e da imagem de profundidade
        detector = RectDetection(self.cv_image, self.cv_depth_image, color)
        
        # A função detect_gate_and_get_error desenha os bounding boxes (retângulos)
        gate_found, _, _, _ = detector.detect_gate_and_get_error()

        processed_filename = os.path.join(self.output_dir, f"detected_{timestamp}.png")
        cv2.imwrite(processed_filename, self.cv_image)
        
        if gate_found:
            print(f"2. Foto com detecção (Portão ENCONTRADO) salva: {processed_filename}")
        else:
            print(f"2. Foto com detecção (Portão NÃO ENCONTRADO) salva: {processed_filename}")
            
        return

if __name__ == '__main__':
    
    # 1. INICIALIZAÇÃO DO ROS 2
    rclpy.init(args=None)
    
    # ZedSubscriber é o nó que lê os tópicos de imagem da ZED
    # Certifique-se de que a classe ZedSubscriber está importada corretamente do mission_base.py
    zed_node = ZedSubscriber()
    
    # 2. INICIALIZAÇÃO DA CLASSE DE FOTO
    picture_taker = Picture(zed_node)
    
    print("Iniciando a captura de fotos. Certifique-se de que a ZED está ligada e o ROS está ativo...")

    try:
        picture_taker.take_and_save_pictures(color='green') # Especifica a cor que será detectada
        
    except Exception as e:
        print(f"Ocorreu um erro durante a execução: {e}")
        
    finally:
        # 3. ENCERRAMENTO
        zed_node.destroy_node()
        rclpy.shutdown()
        print("ROS 2 desligado. Fim do script.")
