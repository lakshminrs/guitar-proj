"""Fine-tune YOLO11-pose on fretboard keypoints and export to ONNX (milestone 3).

  pip install -r requirements-train.txt
  python scripts/train_fretboard.py --data data/fretboard.yaml --epochs 100

Label ~1,500 frames in CVAT or Roboflow with one box per neck ("neck") and the 12 keypoints
listed in data/fretboard.yaml (export as "YOLO pose"). Record yourself + 3-5 friends on
different guitars, lighting and angles. Set FRETBOARD_MODEL_PATH to the exported .onnx.
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fretboard.yaml")
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz,
        # spec augmentations: flips, brightness, blur (ultralytics applies albumentations blur
        # when installed) and occlusion (random erasing ~ hand-shaped masks)
        fliplr=0.5, hsv_v=0.5, hsv_s=0.5, degrees=10, perspective=0.0005, erasing=0.4,
    )
    path = model.export(format="onnx", imgsz=args.imgsz, opset=17, simplify=True)
    print("exported:", path, "\nexport FRETBOARD_MODEL_PATH=" + str(path))


if __name__ == "__main__":
    main()
