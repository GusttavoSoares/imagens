#!/usr/bin/env python3
from enum import Enum
from numpy import np
from dronekit import connect, Vehicle, VehicleMode
from dronekit_utils import arm_and_takeoff, send_velocity_body_offset_ned, goto_position_body_offset_ned
import cv2
import time

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

def find_biggest_err(Image, lower = np.array([127,52,0]), upper = np.array([169,255,255])):
    hsv = cv2.cvtColor(Image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    cnts, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > 1000]

    if len(cnts) <= 0: return None, None
    cntsSorted = sorted(cnts, key=lambda x: cv2.contourArea(x))
    #print(cv2.contourArea(cddnt))
    m = cv2.moments(cnts[0])
    
    if m["m00"] == 0:
        return 0, 0
    x = int(m["m10"]//m["m00"])
    y = int(m["m01"]//m["m00"])

    img_h, img_w, _ = Image.shape

    # diferença entre o centro da base e o centro da imagem da camera 
    return x - img_w/2, img_h/2 - y

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
        
        value = max(min(2, value), -2)

        return value

class States(Enum):

    INIT = 0
    TAKEOFF = 1
    SURVEY = 2
    CENTRALIZE = 3
    LAND = 4
    END = 5

class Mission:
    def __init__(self, vehicle : connect, pid_x : pid, pid_y : pid) -> None:
        self.vehicle = vehicle
        vehicle.home_location = vehicle.location.global_frame

        if DEBUG: print(f"Updated home location to: {vehicle.location.global_frame}")

        self.state = States.INIT
        self.pid_x = pid_x
        self.pid_y = pid_y
        self.capture = cv2.VideoCapture(-1) # talvez precise mudar o número

        if not self.capture.isOpened(): 
            raise RuntimeError("erro camera")

        self.frame = None
        self.running = True

    def update_frame(self):
        ret, frame = self.capture.read()

        if not ret:
            return    

        (h, w) = frame.shape[:2]

        new_width = 640
        new_height = int((new_width / w) * h)
        resized_image = cv2.resize(frame, (new_width, new_height))
        self.frame = resized_image

    def run(self):
        while self.running:
            self.update_frame()
            if DEBUG: print(f"Current state: {self.state}")
            if self.state == States.INIT or self.state == States.TAKEOFF:

                arm_and_takeoff(SURVEY_ALT, self.vehicle)
                goto_position_body_offset_ned(2, 0, 0, vehicle=self.vehicle)
                print(f"Going to SURVEY ({States.SURVEY}) state...")
                self.state = States.SURVEY
            
            if self.state == States.SURVEY:
                if self.frame is None: return False
                err_x, err_y = find_biggest_err(self.frame, COLOR_MASKS[COLOR_TO_LAND]["lower"],COLOR_MASKS[COLOR_TO_LAND]["upper"])
                    
                if err_x is not None:
                    send_velocity_body_offset_ned(self.vehicle, 0, 0, 0)
                    time.sleep(1)
                    err_x, err_y = find_biggest_err(self.frame, COLOR_MASKS[COLOR_TO_LAND]["lower"],COLOR_MASKS[COLOR_TO_LAND]["upper"])
                    if err_x is not None:
                        if abs(err_x) > 0:
                            self.state = States.CENTRALIZE
                            return False
                else:
                    self.state = States.LAND
                    return False

            elif self.state == States.CENTRALIZE:
                    self.pid_x.refresh()
                    self.pid_y.refresh()

                    err_x, err_y = find_biggest_err(Image=self.frame)

                    last_time = time.time()
                    while(time.time() - last_time < TIMEOUT):
                        try:
                            err_x, err_y = find_biggest_err(Image=self.frame, lower=COLOR_MASKS[COLOR_TO_LAND]["lower"], upper=COLOR_MASKS[COLOR_TO_LAND]["upper"])
                            
                            if not err_x:
                                time.sleep(0.01)
                                continue

                            print(f"x:{err_x} y:{err_y}")
                            if abs(err_x) < 40 and abs(err_y) < 40:
                                self.state = States.LAND
                                return False

                            value_x, value_y = self.pid_x.update(err_x), self.pid_y.update(err_y)

                            send_velocity_body_offset_ned(self.vehicle, value_y, value_x, 0.03)

                            last_time = time.time()

                            time.sleep(0.01)
                        except KeyboardInterrupt:
                            self.vehicle.mode = VehicleMode("LAND")
                            break

                    print("Failed to approach :(")

            elif self.state == States.LAND:
                self.vehicle.mode = VehicleMode("LAND")
                time.sleep(15)
                
                self.state = States.END
            
            elif self.state == States.END:
                self.running = False
                self.capture.release()
                cv2.destroyAllWindows()
                return True

            return False

def main(args=None):
    vehicle = connect("/dev/ttyACM0")
    print("Inciando missão")

    mission = Mission(vehicle, pid(0.001, 0, 0), pid(0.001, 0, 0))

    finished = False
    while vehicle.is_armable:
        if finished:
            print('MISSION COMPLETED')
            break
        try:
            finished = mission.run()
        except KeyboardInterrupt:
            print('User shutdown')
            vehicle.mode = VehicleMode("LAND")
            break

        except Exception as e:
            print(e)
            vehicle.mode = VehicleMode("LAND")
            break

        time.sleep(0.01)

if __name__ == '__main__':
    main()