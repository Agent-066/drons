import torch
from ultralytics import YOLO
import cv2


class YOLORecognizer:
    def __init__(self, weights_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Загрузка YOLO модели из .pt файла"""
        self.model = YOLO(weights_path)
        self.device = device
        self.model.to(device)

    def predict(self, image_path, conf_threshold=0.86):
        """
        Распознавание объектов на изображении

        Returns:
            список обнаруженных объектов с координатами, классами и уверенностью
        """
        results = self.model(image_path, conf=conf_threshold, device=self.device)
        return results[0]  # Первое изображение в батче

    def predict_with_details(self, image_path, conf_threshold=0.86):
        """Получение детальной информации об объектах"""
        result = self.predict(image_path, conf_threshold)

        objects = []
        if result.boxes is not None:
            for box in result.boxes:
                objects.append({
                    'class': int(box.cls[0]),
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                })

        return {
            'image_path': image_path,
            'objects': objects,
            'num_objects': len(objects),
            'names': self.model.names
        }

    def visualize(self, image_path, conf_threshold=0.86, save_path='result.jpg'):
        """Визуализация результатов на изображении"""
        results = self.model(image_path, conf=conf_threshold, device=self.device)
        annotated = results[0].plot()
        cv2.imwrite(save_path, annotated)
        return annotated


# Использование
def main():
    WEIGHTS_PATH = "train-12/weights/best.pt"  # Ваш путь
    IMAGE_PATH = "first.jpg"

    recognizer = YOLORecognizer(weights_path=WEIGHTS_PATH)
    result = recognizer.predict_with_details(IMAGE_PATH)

    # Визуализация
    recognizer.visualize(IMAGE_PATH, save_path='current.jpg')


if __name__ == "__main__":
    main()