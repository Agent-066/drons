import torch
from ultralytics import YOLO

class YOLORecognizer:
    def __init__(self, weights_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Инициализация модели происходит ОДИН раз при запуске программы"""
        self.model = YOLO(weights_path)
        self.device = device
        self.model.to(device)

    def process_frame(self, frame, conf_threshold=0.85):
        """
        Принимает кадр из памяти (OpenCV bgr matrix) и возвращает размеченный кадр
        """
        # Передаем frame (массив numpy) напрямую в YOLO вместо пути к файлу
        results = self.model(frame, conf=conf_threshold, device=self.device, verbose=False)

        # Получаем отрисованный кадр с рамками YOLO прямо в памяти
        annotated_frame = results[0].plot()

        return annotated_frame

    def get_details(self, frame, conf_threshold=0.85):
        results = self.model(frame, conf=conf_threshold, device=self.device, verbose=False)
        result = results[0]

        objects = []
        if result.boxes is not None:
            for box in result.boxes:
                objects.append({
                    'class': int(box.cls[0]),
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].tolist()
                })
        return objects


recognizer = YOLORecognizer(weights_path="train-12/weights/best.pt")


def process_frame(frame):
    return recognizer.process_frame(frame)