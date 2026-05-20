import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np


class NeuralNetwork(nn.Module):
    """Пример архитектуры нейросети (настройте под вашу модель)"""

    def __init__(self, num_classes=1):
        super(NeuralNetwork, self).__init__()
        # Пример простой CNN - ЗАМЕНИТЕ на вашу архитектуру!
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),  # Размер зависит от входного изображения
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ImageRecognizer:
    def __init__(self, weights_path, model_architecture=None, num_classes=1,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Инициализация распознавателя изображений

        Args:
            weights_path: путь к файлу .pt с весами
            model_architecture: архитектура модели (если None, используется стандартная)
            num_classes: количество классов для распознавания
            device: устройство (cuda/cpu)
        """
        self.device = device
        print(f"Используется устройство: {self.device}")

        # Загрузка или создание модели
        if model_architecture:
            self.model = model_architecture
        else:
            self.model = NeuralNetwork(num_classes=num_classes)

        # Загрузка весов
        self.load_weights(weights_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Стандартные трансформации для изображений
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Измените размер под вашу модель
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_weights(self, weights_path):
        """Загрузка весов из .pt файла"""
        try:
            checkpoint = torch.load(weights_path, map_location=self.device)

            # Обработка разных форматов сохранения
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)

            print(f"Весы успешно загружены из {weights_path}")
        except Exception as e:
            print(f"Ошибка загрузки весов: {e}")
            raise

    def preprocess_image(self, image_path):
        """Предобработка изображения"""
        try:
            # Загрузка изображения
            if isinstance(image_path, str):
                image = Image.open(image_path).convert('RGB')
            elif isinstance(image_path, Image.Image):
                image = image_path
            else:
                raise ValueError("image_path должен быть строкой пути или PIL Image")

            # Применение трансформаций
            image_tensor = self.transform(image)
            return image_tensor.unsqueeze(0)  # Добавляем batch dimension

        except Exception as e:
            print(f"Ошибка обработки изображения: {e}")
            raise

    def predict(self, image_path, return_probs=False):
        """
        Распознавание изображения

        Args:
            image_path: путь к изображению или PIL Image
            return_probs: возвращать ли вероятности всех классов

        Returns:
            predictions: предсказанный класс или массив вероятностей
        """
        with torch.no_grad():
            # Предобработка
            input_tensor = self.preprocess_image(image_path)
            input_tensor = input_tensor.to(self.device)

            # Инференс
            outputs = self.model(input_tensor)

            # Softmax для вероятностей
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            if return_probs:
                return probabilities.cpu().numpy()[0]
            else:
                predicted_class = torch.argmax(probabilities, dim=1)
                return predicted_class.cpu().numpy()[0]

    def predict_with_details(self, image_path, class_names=None):
        """Распознавание с детальными результатами"""
        probabilities = self.predict(image_path, return_probs=True)
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]

        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': probabilities
        }

        if class_names:
            result['predicted_label'] = class_names[predicted_class]

        return result


# Пример использования
def main():
    # Укажите путь к вашему .pt файлу
    WEIGHTS_PATH = "runs/detect/train-12/weights/best.pt"  # Замените на ваш путь
    IMAGE_PATH = "datasets/valid/images/frame-97-_jpg.rf.4d8fa5a49bca7171844201909eb97a0b.jpg"  # Замените на путь к изображению

    # Создание распознавателя
    recognizer = ImageRecognizer(
        weights_path=WEIGHTS_PATH,
        num_classes=1  # Укажите количество классов вашей модели
    )

    # Распознавание изображения
    try:
        result = recognizer.predict_with_details(IMAGE_PATH)
        print(f"Предсказанный класс: {result['predicted_class']}")
        print(f"Уверенность: {result['confidence']:.4f}")

        # Вывод топ-3 предсказаний
        probs = result['all_probabilities']
        top3 = np.argsort(probs)[-3:][::-1]
        print("\nТоп-3 предсказания:")
        for i, cls in enumerate(top3):
            print(f"  {i + 1}. Класс {cls}: {probs[cls]:.4f}")

    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()