import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',  
            self.image_callback,
            10
        )
        self.image_count = 0
        self.save_interval = 1.0 
        self.last_save_time = self.get_clock().now()
        self.output_dir = os.path.expanduser('~/captured_images')

        os.makedirs(self.output_dir, exist_ok=True)
        self.get_logger().info(f'Salvando imagens em: {self.output_dir}')

    def image_callback(self, msg):
        now = self.get_clock().now()
        if (now - self.last_save_time).nanoseconds * 1e-9 < self.save_interval:
            return

        # Converte a imagem ROS para OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Nome do arquivo com timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(self.output_dir, f'image_{timestamp}.jpg')

        # Salva a imagem
        cv2.imwrite(filename, cv_image)
        self.image_count += 1
        self.last_save_time = now

        self.get_logger().info(f'[{self.image_count}] Imagem salva: {filename}')

def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Encerrando nó de captura...')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
