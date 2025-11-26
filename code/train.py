from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    # Train the model
    results = model.train(
        data="data/data.yaml",
        epochs=1,
        imgsz=640,
        batch=8,
        device="cpu",
        workers=0
    )

    # ✅ Training automatically saves best.pt
    print("✅ Training completed. Best model saved at:", results.save_dir + "/weights/best.pt")

    # ✅ Optional: explicitly export model to another name
    model.export(format="pt", name="cricket_yolov8n_finetuned")

if __name__ == "__main__":
    main()
