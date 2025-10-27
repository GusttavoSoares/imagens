# --- teste_camera.py ---
import rclpy
from rclpy.node import Node
from mission_base import ZedSubscriber
from rect_detection import RectDetection
import cv2
import time


class CameraTester(Node):
    def __init__(self, test_type=1):
        """
        test_type: 1 = testar portão, 2 = testar quadrado
        """
        super().__init__('camera_tester')
        self.zed_node = ZedSubscriber()
        self.get_logger().info("Esperando frames da câmera ZED...")
        self.test_type = test_type

    def run(self):
        """Loop contínuo que decide qual teste executar."""
        try:
            while rclpy.ok():
                # Processa callbacks do ROS2 e atualiza o frame mais recente
                rclpy.spin_once(self.zed_node, timeout_sec=0.05)
                frame = self.zed_node.frame
                frame2 = self.zed_node.depth_frame
                if frame is None:
                    continue

                detector = RectDetection(frame, frame2, 'green')

                if self.test_type == 1:
                    self.test_gate(detector)
                elif self.test_type == 2:
                    self.test_square(detector)
                else:
                    self.get_logger().warn("Teste inválido selecionado!")
                    break

                time.sleep(0.05)

        except KeyboardInterrupt:
            self.get_logger().info("Execução interrompida pelo usuário.")
        finally:
            self.get_logger().info("Encerrando processamento de câmera.")

    def test_gate(self, detector):
        """Teste separado de detecção de portão."""
        gate_found, error_x, error_y, color_detected = detector.detect_gate_and_get_error()
        if gate_found:
            self.get_logger().info(
                f"✅ Portão detectado — Erro X: {error_x:.2f}, Erro Y: {error_y:.2f}, Cor detectada: {color_detected}"
            )
        else:
            self.get_logger().info(
                f"❌ Nenhum portão detectado. Cor visível: {color_detected}"
            )

    def test_square(self, detector):
        """Teste separado de detecção de quadrado."""
        square_found, error_x_sq, x, w = detector.detect_square()
        if square_found:
            # Calcula o centro vertical e erro vertical
            img_h, img_w, _ = detector.image.shape

            self.get_logger().info(
                f"🟩 Quadrado detectado — Erro X: {error_x_sq:.2f}, "
                f"Coordenadas: x={x}, w={w}"
            )
        else:
            self.get_logger().info("❌ Nenhum quadrado detectado.")


def main(args=None):
    rclpy.init(args=args)
    
    choice = input("Escolha o teste: 1 = Portão, 2 = Quadrado: ")
    try:
        test_type = int(choice)
    except ValueError:
        test_type = 1

    tester = CameraTester(test_type=test_type)
    try:
        tester.run()
    finally:
        tester.destroy_node()
        tester.zed_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
