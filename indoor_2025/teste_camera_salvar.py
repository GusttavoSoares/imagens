import rclpy
from rclpy.node import Node
from mission_base import ZedSubscriber
from rect_detection import RectDetection
import cv2
import time
import os


class CameraTester(Node):
    def __init__(self):
        super().__init__('camera_tester')

        # Inicializa o nó assinante da ZED (definido em mission_base.py)
        self.zed_node = ZedSubscriber()
        self.get_logger().info("Esperando frames da câmera ZED...")

        # Cria pasta para salvar as imagens detectadas, se não existir
        self.save_dir = "detected_frames"
        os.makedirs(self.save_dir, exist_ok=True)

    def run(self):
        """Loop contínuo para processar imagens da ZED em tempo real."""
        try:
            while rclpy.ok():
                # Processa callbacks do ROS2 e atualiza o frame mais recente
                rclpy.spin_once(self.zed_node, timeout_sec=0.05)

                frame = self.zed_node.frame
                if frame is None:
                    continue

                # Cria o detector e processa o frame atual
                detector = RectDetection(frame, 'green')
                gate_found, error_x, error_y, color_detected = detector.detect_gate_and_get_error()

                timestamp = time.strftime("%Y%m%d_%H%M%S")

                # Exibe informações no terminal
                if gate_found:
                    self.get_logger().info(
                        f"✅ Portão detectado — Erro X: {error_x:.2f}, Erro Y: {error_y:.2f}, Cor detectada: {color_detected}"
                    )

                    # Salva imagem processada com bounding boxes e centro
                    filename = os.path.join(self.save_dir, f"gate_{timestamp}.jpg")
                    cv2.imwrite(filename, frame)
                    self.get_logger().info(f"📸 Imagem salva em: {filename}")

                else:
                    self.get_logger().info(
                        f"❌ Nenhum portão detectado. Cor visível: {color_detected}"
                    )

                    filename = os.path.join(self.save_dir, f"gate_{timestamp}.jpg")
                    cv2.imwrite(filename, frame)
                

        except KeyboardInterrupt:
            self.get_logger().info("Encerrando captura da ZED...")
        finally:
            self.destroy_node()
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = CameraTester()
    node.run()


if __name__ == '__main__':
    main()
