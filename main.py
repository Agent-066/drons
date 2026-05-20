from ultralytics import YOLO

if __name__ == '__main__':
    # Load a pretrained YOLO26n model
    model = YOLO("yolo11n.pt")

    # Train the model on the COCO8 datasets for 100 epochs
    train_results = model.train(
        data="C:/Users/user/PycharmProjects/PythonProject/datasets/data.yaml",  # Path to datasets configuration file
        batch=4,
        epochs=50,
        # Number of training epochs
        imgsz=640,  # Image size for training
        device="cuda"
        # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)