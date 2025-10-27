#!/usr/bin/env python3
import rclpy
# Importa a classe Mav do seu arquivo de biblioteca
from communication2 import Mav 
import time
import numpy as np 

def main(args=None):
    # Inicializa o contexto ROS 2
    rclpy.init(args=args)
    
    # 1. Instancia o objeto Mav, que é o seu nó ROS 2
    mav = Mav() 
    
    # Definir vértices do quadrado em coordenadas NED (x=Norte, y=Leste, z=-Altura)
    side = 1.2
    altitude = 1.0 
    
    # Waypoints em coordenadas locais (x, y, z)
    # Assumindo que mav.takeoff leva para z=altitude
    waypoints = [
        (0.0, 0.0, altitude),   # Ponto inicial (já na altura)
        (side, 0.0, altitude),  # 1. Norte
        (side, side, altitude), # 2. Nordeste
        (0.0, side, altitude),  # 3. Leste
        (0.0, 0.0, altitude)   # 4. Volta ao início
    ]
    
    # Duração em cada waypoint
    hold_time = 5 

    try:
        mav.get_logger().info("Iniciando missão. Tentando decolar para 1.0m...")
        
        time.sleep(1) # Aguarda o ROS e Mavros inicializarem
        
        # 2. Decolagem
        # O método takeoff já garante GUIDED, ARM e envia o primeiro setpoint de altura.
        if not mav.takeoff(height=altitude): 
             mav.get_logger().error("Falha na decolagem. Encerrando o nó.")
             return
        
        mav.get_logger().info("Decolagem concluída. Iniciando sequência de waypoints.")

        # 3. Movimentação (Quadrado em Posição)
        for i, (x, y, z) in enumerate(waypoints):
            
            # Envia o setpoint de posição
            mav.goto_relative(x=x, y=y, z=z) 
            
            mav.get_logger().info(f"Enviando waypoint {i}: x={x}, y={y}, z={z}")
            
            start_time = time.time()
            # Loop para garantir que o setpoint seja enviado repetidamente (necessário no MAVROS)
            while time.time() - start_time < hold_time:
                # O mav.goto já atualiza o mav.goal_pose, apenas re-publicamos.
                mav.publish_pose(mav.goal_pose) 
                
                # Processa callbacks
                rclpy.spin_once(mav, timeout_sec=0.01)
                time.sleep(0.05) 
                
        # 4. Pouso
        mav.get_logger().info("Sequência de posição finalizada. Pousando...")
        
        if mav.land():
            mav.get_logger().info("Comando de pouso enviado e drone desarmado.")
        else:
            mav.get_logger().error("Falha ao enviar comando de pouso/desarmar.")
        
        time.sleep(5) 

    except Exception as e:
        mav.get_logger().error(f"Um erro ocorreu durante a missão: {e}")
        # Tentativa de pouso de segurança
        try:
             mav.land()
        except:
             pass
    
    finally:
        # Encerramento: Destrói o nó e desliga o ROS 2
        mav.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()