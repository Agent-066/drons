from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")

    train_results = model.train(
        data="C:/Users/user/PycharmProjects/PythonProject/datasets/data.yaml",  # Path to datasets configuration file
        batch=4,
        epochs=50,
        imgsz=640,  # Image size for training
        device="cuda"
        # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)