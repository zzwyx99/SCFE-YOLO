from pathlib import Path

from custom_blocksAFYOLO import register_custom_modules


register_custom_modules()

from ultralytics import YOLO





def main():
    model = YOLO('configs/AF-YOLO.yaml')
    model.train(data="DOTA.yaml", epochs=500, batch=2, imgsz=640, patience=50, deterministic=False, pretrained=False,resume=True,name="AF-YOLO")
if __name__ == "__main__":
    main()
