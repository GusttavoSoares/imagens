#!/usr/bin/env python3

import rclpy
import numpy as np
import time

from communication2 import Mav 
from std_msgs.msg import String, Bool
from enum import Enum
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from mission_base import ZedSubscriber

from mission_1 import Mission_1
from mission_2 import Mission_2
from mission_3 import Mission_3
from mission_4 import Mission_4

ALTITUDE = 1.25

class States(Enum): 
    TAKEOFF = 0
    MISSION_1 = 1
    MISSION_2 = 2
    MISSION_3 = 3
    MISSION_4 = 4
    TRAVELLING = 5
    DIVERTING = 6
    LAND = 7

class Mission:
    done_mission_1 = False

    def __init__(self, mav: Mav, velocity: int, zed_node: ZedSubscriber) -> None:
        self.mission_1 = Mission_1(mav, zed_node)
        self.mission_2 = Mission_2()
        self.mission_3 = Mission_3()
        self.mission_4 = Mission_4(mav, zed_node)

        self.zed_node = zed_node

        self.running = True
        self.mav = mav
        self.state = States.TAKEOFF      
        self.velocity = velocity

    def state_machine(self) -> bool:
        print("Entrou no state machine")

        # ======== TAKEOFF ========
        if self.state == States.TAKEOFF:
            self.mav.get_logger().info("Tentando decolar...")

            if not self.mav.takeoff(height=ALTITUDE): 
                self.mav.get_logger().error("Takeoff error")
                self.state = States.LAND  # força pouso em caso de erro
                return

            # Mantém callbacks e espera estabilizar
            start_time = time.time()
            while time.time() - start_time < 6:
                rclpy.spin_once(self.mav, timeout_sec=0.01)
                rclpy.spin_once(self.zed_node, timeout_sec=0.01)
                time.sleep(0.05)

            self.mav.get_logger().info("Takeoff done. Indo para Mission 1.")
            self.state = States.MISSION_1
            return  # <-- sai do método, próxima iteração entra no bloco MISSION_1

        # ======== MISSION 1 ========
        elif self.state == States.MISSION_1:
            Mission.done_mission_1, problem = self.mission_1.run()
            
            if problem:
                self.state = States.LAND
                return

            if Mission.done_mission_1:
                self.mav.get_logger().info("Mission 1 done. Indo para Travelling.")
                time.sleep(5)
                self.state = States.TRAVELLING
                return

        # ======== MISSION 2 ========
        elif self.state == States.MISSION_2:
            pass

        # ======== LAND ========
        elif self.state == States.LAND:
            self.mav.get_logger().info("LAND")
            self.mav.land()
            self.running = False
            return



    
if __name__ == '__main__':
    
    rclpy.init(args=None)
    zed_node = ZedSubscriber()
    mav = Mav(debug=True)
    mission = Mission(mav=mav, velocity=0.25, zed_node=zed_node)
    time.sleep(5)
    
    mav.get_logger().info("Mission node started")

    try:
        while mission.running:
            rclpy.spin_once(mav, timeout_sec=0.01)
            rclpy.spin_once(zed_node, timeout_sec=0.01)
            print(mav.pose.position.x)
            print(mav.pose.position.y)
            print(mav.pose.position.z)

            mission.state_machine()
            time.sleep(0.01)

    except KeyboardInterrupt:
        print('\nShutdown solicitado pelo usuário (Ctrl+C).')
        mav.land()

    except RuntimeError as e:
        # Captura erros como o erro da câmera
        print(f"ERRO CRÍTICO: {e}")

    except Exception as e:
        # Captura outras exceções durante a execução
        print(f"ERRO INESPERADO: {e}")

    finally:
        mav.destroy_node()
        zed_node.destroy_node()
        rclpy.shutdown()
        print("ROS 2 desligado.")
