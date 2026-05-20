import time

import socket
import cv2
from pioneer_sdk import *

import Ai


def main():
    cam = Camera()

    while True:
        try:
            frame = cam.get_cv_frame()
            if frame is not None:
                cv2.imwrite('first.jpg', frame)
                firstimg = cv2.imread('first.jpg')
                Ai.main()
                Aiimg = cv2.imread('current.jpg')
                aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
                parameters = cv2.aruco.DetectorParameters()
                detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
                corners, ids, rejected = detector.detectMarkers(firstimg)
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(Aiimg, corners, ids)
                cv2.imshow('Pioneer', Aiimg)
            if cv2.waitKey(1) == ord('q'):
                break
        except socket.timeout:
            print("Связь потеряна (Таймаут). Повторное подключение через 1 секунду...")
            time.sleep(1)
            continue
cv2.destroyAllWindows()


if __name__ == '__main__':
    main()