import rclpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.node import Node
import cv2

class ZedSubscriber(Node):
    def __init__(self):
        super().__init__('zed_subscriber')
        self.bridge = CvBridge()
        self.frame = None
        self.depth_frame = None

        # Cria a inscrição no tópico da ZED
        self.subscription_cam = self.create_subscription(
            Image,
            '/zed/zed_node/left/image_rect_color',  # note o / no início
            self.listener_callback,
            10
        )

        self.subscription_depht = self.create_subscription(
            Image,
            'zed/zed_node/depth/depth_registered',
            self.depth_callback,
            10
        )
    
    def listener_callback(self, msg):
        """Callback para a imagem RGB."""
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Opcional: redimensionar ou processar
            # self.frame = cv2.resize(self.frame, (640, 360))
        except Exception as e:
            self.get_logger().error(f"Erro ao converter imagem RGB: {e}")

    

    def depth_callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            self.depth_frame = depth_image
        except Exception as e:
            self.get_logger().error(f"Erro ao converter depth: {e}")

