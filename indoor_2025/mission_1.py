import rclpy
import numpy as np
import cv2
import time

from enum import Enum
from communication2 import Mav 
from rect_detection import RectDetection

class PID:
    def __init__(self, kp = 0.0035, kd = 0.0, ki = 0.00000):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.last_err = 0
        self.last_sum_err = 0

    def refresh(self):
        self.last_err = 0
        self.last_sum_err = 0

    def update(self, err):
        value = self.kp * err + (err - self.last_err)*self.kd + self.last_sum_err * self.ki
        self.last_err = err
        self.last_sum_err += err
        value = max(min(2, value), -2)
        return value
    
class States(Enum):
    ADJUSTMENT = 0
    SIDE_FLY = 1
    CENTRALIZE = 2
    PASS = 3
    FINISHED = 4
    ERROR = 5

class Mission_1:
    def __init__(self, mav: Mav, zed_node) -> None:
        self.mav = mav
        self.mav.get_logger().info("Mission 1 created")
        self.state = States.ADJUSTMENT
        self.zed_node = zed_node
        self.cv_image = None
        self.cv_depth_image = None
        self.offset = 10
        self.alt = 1


        self.pid_sideways = PID(kp=0.0035, kd=0.000, ki=0.0)
        
        self.sideways_velocity_search = 0.2
        self.problem = False
        self.running = True

    # --- ALTERAÇÃO 1: Função helper para atualizar o frame e rodar o spin do ROS ---
    # Centraliza a lógica de atualização para evitar repetição.
    def update_frame(self):
        """Atualiza os nós do ROS e o frame da câmera."""
        rclpy.spin_once(self.mav, timeout_sec=0)
        rclpy.spin_once(self.zed_node, timeout_sec=0)
        self.cv_image = self.zed_node.frame
        self.cv_depth_image = self.zed_node.depth_frame


    def run(self) -> bool:
        # A atualização inicial do frame é movida para dentro dos loops de estado
        if self.state == States.ADJUSTMENT:
            self.mav.get_logger().info("[ADJUSTMENT] Moving forward until nearest object is 0.4 m away.")
            start_time = time.time()
            timeout = 15.0
            target_distance = 0.8
            move_speed = 0.2

            while time.time() - start_time < timeout:
                self.update_frame()

                if self.cv_depth_image is None:
                    self.mav.get_logger().warn("[ADJUSTMENT] Waiting for depth frame...")
                    time.sleep(0.1)
                    continue
                
                detector = RectDetection(self.cv_image, self.cv_depth_image, 'green')

                reached_target, min_depth = detector.is_at_target_distance(
                    target_distance=target_distance
                )

                if min_depth is not None:
                    self.mav.get_logger().info(f"[ADJUSTMENT] Nearest object: {min_depth:.2f} m")
                else:
                    self.mav.get_logger().warn("[ADJUSTMENT] No valid depth data in cropped image.")
                    time.sleep(0.05)
                    continue

                if reached_target:
                    self.mav.set_vel_relative(0.0, 0.0, 0.0)
                    self.mav.get_logger().info(f"[ADJUSTMENT] Target distance reached (<= {target_distance} m).")
                    time.sleep(1.0)
                    self.state = States.SIDE_FLY
                    break
                else:
                    self.mav.set_vel_relative(forward=move_speed, sideways=0.0, upward=0.0)

                time.sleep(0.05)

            else:
                self.mav.get_logger().error("[ADJUSTMENT] Timeout reached without reaching target distance.")
                self.state = States.ERROR

        if self.state == States.SIDE_FLY:
            start_time = time.time()
            timeout = 180  
            gate_found = False
            print("[INFO] Searching for gate!")

            while time.time() - start_time < timeout:
                # --- ALTERAÇÃO 2: Atualiza o frame a cada iteração do loop ---
                self.update_frame()

                if self.cv_image is None:
                    print("Waiting for frame...")
                    time.sleep(0.1)
                    continue

                detector = RectDetection(self.cv_image, self.cv_depth_image, 'green')
                gate_found, _, _, _= detector.detect_gate_and_get_error()

                if gate_found:
                    self.mav.get_logger().info("[SIDE_FLY] Gate Detected!")
                    self.mav.set_vel_relative(0.0, 0.0, 0.0)
                    time.sleep(0.5)
                    self.state = States.CENTRALIZE
                    break  

                self.mav.set_vel_relative(0.0, -self.sideways_velocity_search, 0.0)
                time.sleep(0.1)

            if not gate_found:
                self.mav.get_logger().error("[SIDE_FLY] Timeout! Gate not found.")
                self.mav.set_vel_relative(0.0, 0.0, 0.0)
                self.state = States.ERROR
        
        elif self.state == States.CENTRALIZE:
            self.mav.get_logger().info("[CENTRALIZE] Starting horizontal centralization with PID.")
            self.pid_sideways.refresh()

            start_time = time.time()
            timeout = 20.0
            centralized = False
            lost_counter = 0
            lost_threshold = 10
            while time.time() - start_time < timeout:
                # --- ALTERAÇÃO 3: Atualiza o frame também neste loop ---
                self.update_frame()
                print('memes memes memes')

                if self.cv_image is None:
                    print("Lost frame during centralization...")
                    time.sleep(0.1)
                    continue

                detector = RectDetection(self.cv_image, self.cv_depth_image, 'green')
                gate_found, err_x, _, _= detector.detect_gate_and_get_error() 

                if not gate_found:
                    lost_counter += 1
                    
                    self.mav.get_logger().warn("[CENTRALIZE] Lost sight of the gate!")
                    self.mav.set_vel_relative(0.0, 0.0, 0.0)
                    time.sleep(1.0)
                    self.mav.set_vel_relative(self.alt*(-0.1), self.alt*0.1, 0.0)
                    time.sleep(1.0)
                    self.alt *= -1
                    if lost_counter >= lost_threshold:
                        self.mav.get_logger().error("Deu ruim. Muitos erros")
                        self.state = States.ERROR
                        break
                    continue
                else:
                    lost_counter = 0

                err_corrected = err_x - self.offset

                if abs(err_corrected) < 20: 
                    self.mav.set_vel(0.0, 0.0, 0.0)
                    time.sleep(5.0)
                    self.mav.get_logger().info("[CENTRALIZE] Target successfully centered horizontally.")
                    centralized = True
                    self.state = States.PASS
                    break

                vel_sideways = self.pid_sideways.update(err_corrected)

                self.mav.set_vel_relative(forward=0.0, sideways=float(vel_sideways), upward=0.0)
                self.mav.get_logger().info(f"PID: X_err={err_x:.1f} → V_side={vel_sideways:.3f}")
                time.sleep(0.05)
            
            if not centralized:
                self.mav.get_logger().error("[CENTRALIZE] Failed to centralize within the time limit.")
                self.state = States.ERROR
            
        elif self.state == States.PASS:
            self.mav.get_logger().info("Passing through gates")
            start_time = time.time()
            while time.time() - start_time < 5:
                self.mav.set_vel_relative(0.5, 0.0, 0.0)
                time.sleep(0.05)
            
            self.mav.set_vel_relative(0.0, 0.0, 0.0)
            self.state = States.FINISHED

        elif self.state == States.FINISHED:
            self.mav.get_logger().info("Mission 1 finished")
            return True, self.problem
            
        elif self.state == States.ERROR:
            self.mav.get_logger().error("Unexpected error!")
            self.problem = True
            self.running = False
            self.mav.set_vel(0.0, 0.0, 0.0)
            return False, self.problem
                
        return False, self.problem
