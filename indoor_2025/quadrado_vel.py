#!/usr/bin/env python3
import rclpy
from communication2 import Mav # Importa a classe Mav
import numpy as np
import time
import argparse # Importa o módulo para processar argumentos de linha de comando
import sys # Necessário para passar argumentos para o main

def main(args=None):
    # ----------------------------------------
    # 0. PROCESSAMENTO DE ARGUMENTOS
    # ----------------------------------------
    
    # Garantir que sys.argv[1:] seja usado se args for None (execução direta)
    if args is None:
        args = sys.argv[1:]
        
    parser = argparse.ArgumentParser(description='Missão de Quadrado por Velocidade com Decolagem Opcional.')
    
    parser.add_argument(
        '--skip-takeoff', 
        action='store_true', 
        help='Se presente, o drone não fará a decolagem (assume-se que já está no ar e em GUIDED).'
    )
    
    parsed_args = parser.parse_args(args)
    
    # ----------------------------------------
    # 1. INICIALIZAÇÃO ROS 2
    # ----------------------------------------

    rclpy.init(args=None) 
    
    mav = Mav(debug=True) 
    
    # Flag booleana que decide se a decolagem E o pouso serão executados
    do_takeoff_and_land = not parsed_args.skip_takeoff

    # --- CONFIGURAÇÕES DA MISSÃO ---
    x_vel = 0.25
    segment_time = 4 
    altitude = 1.0
    STABILIZATION_TIME = 2.0 # Tempo de espera para o drone estabilizar

    waypoints = [
        (x_vel, 0.0, 0.0),   # 1. Frente (Forward +X)
        (0.0, x_vel, 0.0),  # 2. Esquerda (Sideways -Y)
        (-x_vel, 0.0, 0.0),  # 3. Trás (Forward -X)
        (0.0, -x_vel, 0.0),   # 4. Direita (Sideways +Y)
    ]

    try:
        mav.get_logger().info("Iniciando missão...")
        
        # 1. Inicializa o setpoint (CRÍTICO para o MAVROS/PX4)
        #mav.setpoint_init()
        #time.sleep(1) 
        
        # 2. DECOLAGEM CONDICIONAL E GARANTIA DE MODO
        if do_takeoff_and_land:
            mav.get_logger().info(f"Flag --skip-takeoff AUSENTE. Tentando DECOLAGEM para {altitude}m...")
            if not mav.takeoff(height=altitude):
                 mav.get_logger().error("Falha na decolagem. Encerrando o nó.")
                 return
            
            # NOVO: Pausa para o drone estabilizar na altura alvo antes de qualquer outro comando
            mav.get_logger().info(f"Decolagem concluída. Esperando {STABILIZATION_TIME}s para estabilizar e garantir o controle.")
            time.sleep(STABILIZATION_TIME)
            
            # Reforça o modo GUIDED (4), após a estabilização
            if not mav.change_mode("4"):
                mav.get_logger().error("Falha ao reforçar o modo GUIDED. Encerrando o nó.")
                return
            mav.get_logger().info("Modo GUIDED garantido.")
        else:
            mav.get_logger().info("Flag --skip-takeoff DETECTADA. Pulando decolagem e assumindo modo GUIDED/Armado.")

        mav.get_logger().info(f"Iniciando sequência de velocidade com {x_vel} m/s.")
        
        # 3. MOVIMENTAÇÃO: Sequência de Waypoints (Velocidade Relativa)
        for (vx, vy, vz) in waypoints:
            
            mav.get_logger().info(f"Enviando vel relativa: Forward={vx}, Sideways={vy}")
            
            start_time = time.time()
            
            while time.time() - start_time < segment_time: 
                mav.set_vel_relative(forward=vx, sideways=vy, upward=vz) 
                
                rclpy.spin_once(mav, timeout_sec=0.01) 
                time.sleep(0.05)
                
            mav.set_vel_relative(0.0, 0.0, 0.0) # Parar
            time.sleep(1)
            rclpy.spin_once(mav, timeout_sec=0.1)


        # 4. POUSO CONDICIONAL
        mav.get_logger().info("Sequência de velocidade finalizada.")
        
        if mav.land():
            mav.get_logger().info("Pouso concluído.")
        else:
            mav.get_logger().info("Pouso pulado. O drone permanece no modo e altura atuais.")
        
        time.sleep(5) 

    except Exception as e:
        mav.get_logger().error(f"Um erro ocorreu durante a missão: {e}")
        # Pouso de segurança APENAS se tentou decolar.
        if do_takeoff_and_land:
             try:
                 mav.land()
             except:
                 pass
    
    finally:
        mav.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
