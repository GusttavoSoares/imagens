import rclpy
import math
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped, Twist
from mavros_msgs.srv import SetMode, CommandTOL, CommandBool
from mavros_msgs.msg import PositionTarget
import time

PI = np.pi
HALF_PI = PI/2.0

class Mav(Node):
    """
    Interface with mavros
    """

    def __init__(self, debug : bool = False, simulation: bool = False, lidar_min: float = 0.3, indoor: bool = False, zed: bool = True) -> None:

        #INITIALIZING NODE
        super().__init__("mav")
        self.simulation = simulation
        self.zed = zed
        if self.zed: self.const = -1
        else: self.const = 1
        self.lidar_min_distance = lidar_min

        if debug: self.get_logger().info("started mav")

        #SUBSCRIBERS
        
        if simulation or indoor: self.pose_sub = self.create_subscription(PoseStamped, "/mavros/local_position/pose", self.pose_callback, 10)
        else: self.pose_sub = self.create_subscription(PoseStamped, "/mavros/vision_pose/pose", self.pose_callback, 10)

        #PUBLISHERS
        self.pos_pub = self.create_publisher(PoseStamped, "/mavros/setpoint_position/local", 1)
        self.raw_pub = self.create_publisher(PositionTarget, "/mavros/setpoint_raw/local", 1)
        self.vel_pub = self.create_publisher(Twist, "/mavros/setpoint_velocity/cmd_vel_unstamped", 1)

        #SERVICES
        self.mode_serv = self.create_client(SetMode, '/mavros/set_mode')
        self.arm_serv = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.takeoff_serv = self.create_client(CommandTOL, '/mavros/cmd/takeoff')
        
        #ATTRIBUTES
        self.pose = Pose()
        self.goal_pose = Pose()
        self.debug = debug
        self.mode = int()

    def euler_from_quaternion(self, quaternion):
        """
        Converts quaternion (w in last place) to euler roll, pitch, yaw
        quaternion = [x, y, z, w]
        Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
        """
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw
    
    def quaternion_from_euler(self, yaw, pitch, roll):

        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qx, qy, qz, qw]
    
    def in_between(self, check, center, margin):
        """
        This function checks if a number is in certain interval of other number

        check: number to be compared
        center: reference number
        margin: maximum interval allowed
        """
        return (check <= center + margin and check >= center - margin)
    
    def pose_callback(self, msg : PoseStamped) -> None:
        """
        ROS callback used to get local position PoseStamped messages.
        """
        self.pose = msg.pose
        if not self.simulation: self.pose.position.z += self.lidar_min_distance

    def set_vel(self, vel_x : float=0.0, vel_y : float=0.0, vel_z : float=0.0, ang_x : float=0.0, ang_y : float=0.0, ang_z : float=0.0) -> None:
        """
        Populates a Twist object with velocity information
        """
        twist = Twist()

        twist.linear.x = vel_x
        twist.linear.y = self.const*vel_y
        twist.linear.z = vel_z

        twist.angular.x = ang_x
        twist.angular.y = ang_y
        twist.angular.z = ang_z

        self.vel_pub.publish(twist)

    def set_vel_relative(self, forward: float = 0.0, sideways: float = 0.0, upward:float = 0.0) -> None:

        res = PositionTarget()
        res.header.stamp = self.get_clock().now().to_msg()
        res.header.frame_id = "base_footprint"

        res.velocity.x = forward
        res.velocity.y = self.const*sideways
        res.velocity.z = upward
        res.coordinate_frame = PositionTarget.FRAME_BODY_OFFSET_NED
        res.type_mask = PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ | PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ | PositionTarget.FORCE | PositionTarget.IGNORE_YAW | PositionTarget.IGNORE_YAW_RATE

        self.raw_pub.publish(res)
        rclpy.spin_once(self, timeout_sec=0)

    def publish_pose(self, pose : Pose) -> None:
        """
        Populates a PoseStamped object with pose and publishes it
        """
        stamped = PoseStamped()
        stamped.pose = pose

        self.pos_pub.publish(stamped)

    def goto(self, x=None, y=None, z=None, yaw=None, send_time=None) -> None:
        """
        Sends a Pose message and publishes it as a setpoint (assuming vehicle is in guided mode). Yaw is offset by pi/2.
        If movement in a specifit axis is not provided, assumes that you want to keep the vehicles current axial position.
        Updates the self.goal_pose variable as well
        """

        #if new position on axis is provided use it, otherwise just keep current one
        self.goal_pose.position.x = x if x != None else self.pose.position.x
        self.goal_pose.position.y = self.const*y if y != None else self.pose.position.y
        self.goal_pose.position.z = z if z != None else self.pose.position.z

        #if new yaw was given, use it. Otherwise keep the vehicles current yaw
        if yaw != None:
            quat = self.quaternion_from_euler(0, 0, yaw + HALF_PI)
            self.goal_pose.orientation.x = quat[0]
            self.goal_pose.orientation.y = quat[1]
            self.goal_pose.orientation.z = quat[2]
            self.goal_pose.orientation.w = quat[3]
        else:
            self.goal_pose.orientation.x = self.pose.orientation.x
            self.goal_pose.orientation.y = self.pose.orientation.y
            self.goal_pose.orientation.z = self.pose.orientation.z
            self.goal_pose.orientation.w = self.pose.orientation.w

        if self.debug: self.get_logger().info(f"[GOTO] Sending goto {self.goal_pose}")

        if send_time is not None:

            if self.debug: self.get_logger().info(f"[GOTO] Keep sending goto for {send_time} seconds")

            start_time = time.time()
            while time.time() - start_time < send_time: 
                self.publish_pose(pose=self.goal_pose)
        else:
            self.publish_pose(pose=self.goal_pose)
        self.get_logger().info(f"[GOTO] Finished")

    def goto_relative(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, yaw: float = HALF_PI, send_time=None) -> None:
        """
        Envia um comando de deslocamento (offset) relativo ao corpo do drone, usando PositionTarget.
        x: Offset para Frente (Forward)
        y: Offset para Direita (Sideways)
        z: Offset para Cima (Upward)
        """
        res = PositionTarget()
        res.header.stamp = self.get_clock().now().to_msg()
        res.header.frame_id = "base_footprint"
        
        res.position.x = x
        
        res.position.y = self.const*y 
        res.position.z = z
        res.yaw = yaw 


        res.coordinate_frame = PositionTarget.FRAME_BODY_OFFSET_NED
        
        res.type_mask = PositionTarget.IGNORE_VX | PositionTarget.IGNORE_VY | PositionTarget.IGNORE_VZ | \
                        PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ | \
                        PositionTarget.IGNORE_YAW_RATE

        if send_time is not None:
            if self.debug: self.get_logger().info(f"[GOTO_RELATIVE] Keep sending raw setpoint for {send_time} seconds")
            
            start_time = time.time()
            while time.time() - start_time < send_time:
                self.raw_pub.publish(res) # Publica a PositionTarget
                rclpy.spin_once(self, timeout_sec=0.01)
                time.sleep(0.05)
        else:
            self.raw_pub.publish(res)
        
        self.get_logger().info(f"[GOTO_RELATIVE] Finished publishing raw setpoint.")

    def distance_to_goal(self) -> float:
        """
        Calculates the euclidian distance between current position and goal position
        """
        dx = self.goal_pose.position.x - self.pose.position.x
        dy = self.goal_pose.position.y - self.pose.position.y
        dz = self.goal_pose.position.z - self.pose.position.z

        return math.sqrt((dx * dx) + (dy * dy) + (dz * dz))

    def rotate_control_yaw(self, yaw : float, yaw_rate: float = 0.5) -> None:
        """
        Rotates vehicles yaw. Its using setpoint raw
        """
        angle = PositionTarget()
        angle.header.stamp = self.get_clock().now().to_msg()
        angle.header.frame_id = "base_footprint"

        angle.yaw = yaw
        angle.yaw_rate = yaw_rate

        angle.coordinate_frame = PositionTarget.FRAME_BODY_NED
        angle.position.x = self.pose.position.x
        angle.position.y = self.pose.position.y
        angle.position.z = self.pose.position.z
        angle.type_mask = PositionTarget.IGNORE_VX | PositionTarget.IGNORE_VY | PositionTarget.IGNORE_VZ | PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ

        self.raw_pub.publish(angle)

        if self.debug: self.get_logger().info(f"[ROTATE] Rotating to {yaw} rad, with a rate of {yaw_rate} rad/2")

    def rotate(self, yaw : float) -> None:
        """
        Rotates vehicles yaw. The same as goto but only changes yaw
        """
        self.goto(x=self.pose.position.x, y=self.pose.position.y, z=self.pose.position.z, yaw=yaw)

    def rotate_relative(self, yaw : float) -> None:
        """
        Rotates vehicles yaw in relative of its current position. Using the rotate method
        """

        yaw_current = self.euler_from_quaternion([self.pose.orientation.x, self.pose.orientation.y,
                                                            self.pose.orientation.z, self.pose.orientation.w])[2]

        self.rotate(yaw=(yaw_current + yaw - HALF_PI))

    def arm(self) -> bool:
        """
        Arms throttle (bool=True)
        """
        req = CommandBool.Request()
        req.value = True
        
        future = self.arm_serv.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        if future.result() is not None and future.result().success:
            self.get_logger().info("Drone armado.")
            return True
        else:
            self.get_logger().error("Falha ao armar o drone.")
            return False

    def disarm(self) -> bool:
        """
        Disarms throttle (bool=False)
        """
        req = CommandBool.Request()
        req.value = False
        
        future = self.arm_serv.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        
        if future.result() is not None and future.result().success:
            self.get_logger().info("Drone desarmado.")
            return True
        else:
            self.get_logger().error("Falha ao desarmar o drone.")
            return False

    def takeoff(self, height : float= 1) -> bool:
                    
        rclpy.spin_once(self, timeout_sec=0.1)

        if self.in_between(self.pose.position.z, height, 0.1): 
            self.get_logger().info("Já na altura de decolagem.")
            return True
        
        self.get_logger().info("Serviços prontos. Tentando armar e decolar...")

        if not self.change_mode("4"):
            self.get_logger().error("Falha ao mudar para o modo GUIDED.")
            return False

        if not self.arm():
            self.get_logger().error("Falha ao armar o drone.")
            return False
        
        time.sleep(3.0)
            
        self.get_logger().info("Publicação inicial do setpoint concluída.")

        tol_req = CommandTOL.Request()
        tol_req.altitude = float(height)
        
        future_tol = self.takeoff_serv.call_async(tol_req)

        time.sleep(0.1)

        rclpy.spin_until_future_complete(self, future_tol, timeout_sec=7.0)
                    
        self.get_logger().info("Comando de Takeoff enviado. Assumindo controle de posição...")
                
        self.get_logger().info(f"Decolagem bem-sucedida para {height} metros! 🚀")
        return True

    def change_mode(self, mode : str) -> bool:
        """
        Changes vehicle mode to given string. Some modes include:
        STABILIZE = 0, GUIDED = 4, LAND = 9
        """
        req = SetMode.Request()
        req.custom_mode = mode
        
        future = self.mode_serv.call_async(req)
        
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        if future.result() is not None and future.result().mode_sent:
            self.get_logger().info(f"Modo alterado para: {mode}")
            return True
        else:
            self.get_logger().error(f"Falha ao mudar o modo para: {mode}")
            return False

    def land(self) -> bool:
        """
        Sets mode to land and disarms drone.
        """
        return self.change_mode("9") and self.disarm()
     
