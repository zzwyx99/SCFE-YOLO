from ultralytics import YOLO
from custom_blocks import register_custom_modules
register_custom_modules()
if __name__ == '__main__':
    model = YOLO("configs/SCFE-YOLO.yaml")
    model.train(data="VisDrone.yaml", epochs=500, batch=8
    , imgsz=640, deterministic=False,name="SCFE-YOLO",pretrained=False)
