#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from enum import Enum
import numpy as np 
import cv2
import time

from communication2 import Mav

# --- CONSTANTES ---
DEBUG = True
SURVEY_ALT = 2.0
TIMEOUT = 4

COLOR_TO_LAND = "blue" 

COLOR_MASKS = {
    "purple": {
        "lower": np.array([127, 52, 0]),
        "upper": np.array([169, 255, 255])
    },
    "red": {
        "lower": np.array([0, 100, 100]),
        "upper": np.array([10, 255, 255])
    },
    "green": {
        "lower": np.array([40, 100, 100]),
        "upper": np.array([80, 255, 255])
    },
    "blue": {
        "lower": np.array([100, 100, 100]),
        "upper": np.array([140, 255, 255])
    }
}

# --- FUNÇÕES DE VISÃO ---

def find_biggest_err(Image, lower = np.array([100,100,100]), upper = np.array([140,255,255])):
    hsv = cv2.cvtColor(Image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    cnts, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > 1000]

    if len(cnts) <= 0: return None, None
    
    # Corrigido: o código estava usando cnts[0] em vez do maior contorno. 
    # Assumindo que você quer o maior:
    cntsSorted = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)
    biggest_cnt = cntsSorted[0]

    m = cv2.moments(biggest_cnt)
    
    if m["m00"] == 0:
        return 0, 0
    x = int(m["m10"]//m["m00"])
    y = int(m["m01"]//m["m00"])

    img_h, img_w, _ = Image.shape

    # diferença entre o centro da base e o centro da imagem da camera 
    return x - img_w/2, img_h/2 - y

# --- CLASSE PID ---

class pid:
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
        
        # Limita a velocidade (velocidade máxima +/- 2 m/s)
        value = max(min(2, value), -2)

        return value

# --- CLASSES DE MISSÃO ---

class States(Enum):
    TAKEOFF = 0
    SURVEY = 1
    CENTRALIZE = 2
    LAND = 3
    END = 4

class Mission: # CORRIGIDO: Não herda de Node
    def __init__(self, mav: Mav, pid_x : pid, pid_y : pid) -> None:
        self.mav = mav

        self.state = States.TAKEOFF
        self.pid_x = pid_x
        self.pid_y = pid_y
        self.capture = cv2.VideoCapture(0) # talvez precise mudar o número

        if not self.capture.isOpened(): 
            self.mav.get_logger().error("Erro ao abrir a câmera (cv2.VideoCapture).")
            # A exceção será capturada no bloco try/except da main
            raise RuntimeError("Erro ao abrir a câmera") 

        self.frame = None
        self.running = True

    def update_frame(self):
        ret, frame = self.capture.read()

        if not ret:
            # self.mav.get_logger().warn("Falha ao ler frame da câmera.")
            return    

        (h, w) = frame.shape[:2]

        new_width = 640
        new_height = int((new_width / w) * h)
        resized_image = cv2.resize(frame, (new_width, new_height))
        self.frame = resized_image

    def run(self):
        # Esta função retorna True apenas se a missão foi concluída (END)
        self.update_frame()
        if DEBUG: self.mav.get_logger().info(f"Current state: {self.state.name}")
        
        if self.state == States.TAKEOFF:
            self.mav.get_logger().info(f"Decolando para {SURVEY_ALT}m...")
            if self.mav.takeoff(SURVEY_ALT):
                time.sleep(8) # Dá tempo para estabilizar
                self.state = States.SURVEY
            else:
                self.mav.get_logger().error("Decolagem falhou.")
                self.state = States.LAND

        elif self.state == States.SURVEY:
            if self.frame is None: 
                self.mav.get_logger().warn("Frame da câmera indisponível durante SURVEY.")
                return False
                
            err_x, err_y = find_biggest_err(self.frame, COLOR_MASKS[COLOR_TO_LAND]["lower"],COLOR_MASKS[COLOR_TO_LAND]["upper"])
                
            if err_x is not None:
                self.mav.get_logger().info(f"Alvo encontrado. Erro inicial X:{err_x}, Y:{err_y}")
                self.mav.set_vel(0, 0, 0) # Para o drone
                time.sleep(1)
                
                # Re-verifica após parar (garante que não foi um falso positivo)
                err_x, err_y = find_biggest_err(self.frame, COLOR_MASKS[COLOR_TO_LAND]["lower"],COLOR_MASKS[COLOR_TO_LAND]["upper"])
                
                if err_x is not None:
                    self.state = States.CENTRALIZE
            else:
                self.mav.get_logger().warn("Alvo não encontrado. Tentando pousar (LAND).")
                self.state = States.LAND

        elif self.state == States.CENTRALIZE:
            self.pid_x.refresh()
            self.pid_y.refresh()

            self.mav.get_logger().info("Iniciando centralização.")
            
            start_time = time.time()
            while(time.time() - start_time < TIMEOUT):
                # Processa ROS para não bloquear
                rclpy.spin_once(self.mav, timeout_sec=0) 
                self.update_frame()
                
                try:
                    err_x, err_y = find_biggest_err(Image=self.frame, lower=COLOR_MASKS[COLOR_TO_LAND]["lower"], upper=COLOR_MASKS[COLOR_TO_LAND]["upper"])
                    
                    if err_x is None:
                        self.mav.get_logger().warn("Alvo perdido durante a centralização. Tentando pousar.")
                        self.state = States.LAND
                        return False

                    if abs(err_x) < 40 and abs(err_y) < 40:
                        self.mav.set_vel(0, 0, 0)
                        self.mav.get_logger().info("Alvo centralizado com sucesso.")
                        self.state = States.LAND
                        return False

                    value_x, value_y = self.pid_x.update(err_x), self.pid_y.update(err_y)
                    
                    # CORREÇÃO LÓGICA: set_vel_relative geralmente espera (forward, sideways, upward)
                    # Assumindo que err_x (horizontal) controla sideways (Y) e err_y (vertical) controla forward (X) ou pitch.
                    # Vamos assumir que X é lateral (sideways) e Y é forward (para frente/trás) se a câmera estiver olhando para baixo.
                    # Se err_x positivo = objeto à direita, drone deve ir para a direita (Y positivo).
                    self.mav.set_vel_relative(forward=value_y, sideways=value_x, upward=0.03)
                    
                    self.mav.get_logger().info(f"PID: X={err_x:.1f} → V_Y={value_x:.3f} | Y={err_y:.1f} → V_X={value_y:.3f}")

                    time.sleep(0.01)

                except Exception as e:
                    self.mav.get_logger().error(f"Erro no loop de centralização: {e}")
                    self.state = States.LAND
                    break
            
            if self.state != States.LAND:
                self.mav.get_logger().error("Falha ao centralizar dentro do tempo limite.")
                self.state = States.LAND


        elif self.state == States.LAND:
            self.mav.get_logger().info("Iniciando procedimento de pouso...")
            self.mav.set_vel(0, 0, 0)
            if self.mav.land():
                self.mav.get_logger().info("Comando de pouso enviado. Aguardando desarme.")
            else:
                self.mav.get_logger().error("Falha ao enviar comando de pouso.")
            
            time.sleep(15) # Dá tempo para o pouso ser executado
            self.state = States.END
        
        elif self.state == States.END:
            self.running = False
            self.capture.release()
            cv2.destroyAllWindows()
            return True

        return False

# --- FUNÇÃO PRINCIPAL DE EXECUÇÃO ---

def main(args=None):
    # 1. INICIALIZAÇÃO ROS 2
    rclpy.init(args=args)

    # 2. CRIAÇÃO DO NÓ (Mav é o nó)
    try:
        mav_node = Mav(debug=DEBUG) 
        
    except Exception as e:
        print(f"ERRO: Falha ao criar o nó Mav: {e}")
        rclpy.shutdown()
        return

    print("Iniciando missão")

    # 3. CRIAÇÃO DA MISSÃO
    pid_x = pid(0.003, 0.0, 0.0) # Ajuste os ganhos PID conforme necessário
    pid_y = pid(0.003, 0.0, 0.0)
    mission = Mission(mav_node, pid_x, pid_y) 

    finished = False
    
    try:
        # Loop principal que mantém o programa ativo
        while mission.running:
            # Processa TODAS as callbacks (pose, serviços, etc.) do nó Mav
            rclpy.spin_once(mav_node, timeout_sec=0) 
            
            # Executa a lógica do estado atual da missão
            finished = mission.run()
            
            if finished:
                print('MISSÃO CONCLUÍDA')
                break

            # Pequeno delay para economizar CPU, se o loop for muito rápido
            time.sleep(0.01) 
            
    except KeyboardInterrupt:
        print('\nShutdown solicitado pelo usuário (Ctrl+C).')
    
    except RuntimeError as e:
        # Captura erros como o erro da câmera
        print(f"ERRO CRÍTICO: {e}")

    except Exception as e:
        # Captura outras exceções durante a execução
        print(f"ERRO INESPERADO: {e}")
        
    finally:
        # 4. ENCERRAMENTO
        
        # Garante o pouso de segurança antes de desligar (se não estiver no estado END)
        if mission.state != States.END:
            print('Tentando pouso/desarme de segurança...')
            try:
                mav_node.set_vel(0, 0, 0)
                mav_node.land()
                time.sleep(5) 
            except Exception as e:
                 print(f"Falha no pouso de segurança: {e}")

        # Limpeza final
        mav_node.destroy_node()
        rclpy.shutdown()
        print("ROS 2 desligado.")


if __name__ == '__main__':
    main()