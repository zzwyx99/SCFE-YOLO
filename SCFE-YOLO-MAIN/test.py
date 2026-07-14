from ultralytics import YOLO
from custom_blocks import register_custom_modules
register_custom_modules()
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("SCFEYOLO.pt")
    results = model.val(data="VisDrone.yaml", imgsz=640, name="custom_-noFElaocangku", split="val")
