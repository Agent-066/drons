import socket
import time
import cv2
import Ai
from pioneer_sdk import *

def main():
    cam = Camera()

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    print("Система запущена. Нажмите 'q' для выхода.")

    while True:
        try:
            frame = cam.get_cv_frame()
            if frame is None:
                continue

            ai_frame = Ai.process_frame(frame)

            corners, ids, _ = detector.detectMarkers(frame)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(ai_frame, corners, ids)

            cv2.imshow('Pioneer', ai_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except socket.timeout:
            print("Связь потеряна (Таймаут). Повторное подключение через 1 секунду...")
            time.sleep(1)
        except Exception as e:
            print(f"Ошибка в цикле: {e}")
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
