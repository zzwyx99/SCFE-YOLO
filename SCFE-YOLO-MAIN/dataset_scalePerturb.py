import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


# =========================
# 1. 读写 YOLO 标签
# =========================
def read_yolo_labels(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])
            boxes.append([cls_id, xc, yc, bw, bh])
    return boxes


def write_yolo_labels(label_path, boxes):
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, "w") as f:
        for box in boxes:
            cls_id, xc, yc, bw, bh = box
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


# =========================
# 2. YOLO <-> xyxy
# =========================
def yolo_to_xyxy(box, img_w, img_h):
    cls_id, xc, yc, bw, bh = box
    x_center = xc * img_w
    y_center = yc * img_h
    box_w = bw * img_w
    box_h = bh * img_h

    x1 = x_center - box_w / 2
    y1 = y_center - box_h / 2
    x2 = x_center + box_w / 2
    y2 = y_center + box_h / 2
    return [cls_id, x1, y1, x2, y2]


def xyxy_to_yolo(cls_id, x1, y1, x2, y2, img_w, img_h):
    xc = ((x1 + x2) / 2) / img_w
    yc = ((y1 + y2) / 2) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return [cls_id, xc, yc, bw, bh]


# =========================
# 3. 计算所有 GT 的并集框
# =========================
def get_union_box(gt_boxes, img_w, img_h):
    if len(gt_boxes) == 0:
        return None

    xyxy_boxes = [yolo_to_xyxy(box, img_w, img_h) for box in gt_boxes]
    x1 = min(b[1] for b in xyxy_boxes)
    y1 = min(b[2] for b in xyxy_boxes)
    x2 = max(b[3] for b in xyxy_boxes)
    y2 = max(b[4] for b in xyxy_boxes)
    return [x1, y1, x2, y2]


# =========================
# 4. 计算居中缩放后的贴图变换
# 参考 单个热力图.py 的 make_scaled_black_canvas：
#   - 整图缩小
#   - 再贴到同尺寸黑色画布中央
# =========================
def get_center_paste_transform(img_w, img_h, scale):
    new_w = max(1, int(round(img_w * scale)))
    new_h = max(1, int(round(img_h * scale)))

    offset_x = (img_w - new_w) // 2
    offset_y = (img_h - new_h) // 2

    # resize 后的尺寸经过了 round，标签同步时使用真实缩放比例更稳妥。
    scale_x = new_w / float(img_w)
    scale_y = new_h / float(img_h)

    return new_w, new_h, offset_x, offset_y, scale_x, scale_y


def transform_xyxy_with_center_paste(x1, y1, x2, y2, offset_x, offset_y, scale_x, scale_y):
    nx1 = x1 * scale_x + offset_x
    ny1 = y1 * scale_y + offset_y
    nx2 = x2 * scale_x + offset_x
    ny2 = y2 * scale_y + offset_y
    return nx1, ny1, nx2, ny2


# =========================
# 5. 高视角图像与标签同步变换
# scale < 1 -> 更高视角（目标更小）
# 逻辑：
#   - 整幅图按 scale 缩小
#   - 放置到原大小黑色画布中央
#   - 新增区域用黑色填充
# =========================
def make_high_view_image_and_labels(img, gt_boxes, scale):
    """
    img: RGB image, shape(H, W, 3)
    gt_boxes: YOLO normalized boxes
    scale: 0 < scale < 1, 越小表示视角越高，目标越小
    """
    if not (0 < scale < 1):
        raise ValueError(f"高视角 scale 应满足 0 < scale < 1，但收到: {scale}")

    img_h, img_w = img.shape[:2]
    new_w, new_h, offset_x, offset_y, scale_x, scale_y = get_center_paste_transform(img_w, img_h, scale)

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    high_view_img = np.zeros_like(img, dtype=np.uint8)
    high_view_img[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

    # 同步变换标签
    new_boxes = []
    for box in gt_boxes:
        cls_id, x1, y1, x2, y2 = yolo_to_xyxy(box, img_w, img_h)

        # 标签和图像使用完全相同的缩放尺寸与居中偏移。
        nx1, ny1, nx2, ny2 = transform_xyxy_with_center_paste(
            x1, y1, x2, y2,
            offset_x, offset_y,
            scale_x, scale_y
        )

        # 裁剪到图像边界
        nx1 = max(0.0, min(float(img_w), nx1))
        ny1 = max(0.0, min(float(img_h), ny1))
        nx2 = max(0.0, min(float(img_w), nx2))
        ny2 = max(0.0, min(float(img_h), ny2))

        # 过滤无效框
        if nx2 <= nx1 or ny2 <= ny1:
            continue

        new_box = xyxy_to_yolo(cls_id, nx1, ny1, nx2, ny2, img_w, img_h)
        new_boxes.append(new_box)

    return high_view_img, new_boxes


# =========================
# 6. 处理单个 split
# 比如 train / val / test
# =========================
def process_split(images_dir, labels_dir, out_images_dir, out_labels_dir, scale):
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])

    valid_count = 0
    skip_count = 0

    for image_name in tqdm(image_files, desc=f"{Path(images_dir).name} | x{scale}"):
        image_path = os.path.join(images_dir, image_name)
        label_name = Path(image_name).with_suffix(".txt").name
        label_path = os.path.join(labels_dir, label_name)

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            skip_count += 1
            continue

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt_boxes = read_yolo_labels(label_path)

        high_view_img, new_boxes = make_high_view_image_and_labels(img, gt_boxes, scale)

        save_image_path = os.path.join(out_images_dir, image_name)
        save_label_path = os.path.join(out_labels_dir, label_name)

        cv2.imwrite(save_image_path, cv2.cvtColor(high_view_img, cv2.COLOR_RGB2BGR))
        write_yolo_labels(save_label_path, new_boxes)

        valid_count += 1

    print(f"[Done] {images_dir}")
    print(f"       valid: {valid_count}")
    print(f"       skip : {skip_count}")


# =========================
# 7. 处理整个数据集
# 默认目录结构:
# dataset_root/
#   images/train, images/val, images/test
#   labels/train, labels/val, labels/test
# =========================
def process_dataset(dataset_root, output_parent, scales=(0.75, 0.5, 0.25)):
    """
    scales < 1:
        0.75 -> 轻度高视角
        0.50 -> 中度高视角
        0.25 -> 强高视角
    """
    dataset_root = Path(dataset_root)
    output_parent = Path(output_parent)

    split_names = []
    for split in ["train", "val", "test"]:
        if (dataset_root / "images" / split).exists():
            split_names.append(split)

    if len(split_names) == 0:
        raise ValueError("未找到 images/train 或 images/val 或 images/test 目录，请检查数据集路径。")

    print("检测到 splits:", split_names)

    for scale in scales:
        if not (0 < scale < 1):
            raise ValueError(f"高视角 scale 必须在 (0, 1) 内，当前为 {scale}")

        out_root = output_parent / f"{dataset_root.name}_highview_x{scale}"

        for split in split_names:
            images_dir = dataset_root / "images" / split
            labels_dir = dataset_root / "labels" / split

            out_images_dir = out_root / "images" / split
            out_labels_dir = out_root / "labels" / split

            if not images_dir.exists():
                continue

            if not labels_dir.exists():
                os.makedirs(out_labels_dir, exist_ok=True)

            print(f"\n========== high-view scale x{scale} | split={split} ==========")
            process_split(
                images_dir=str(images_dir),
                labels_dir=str(labels_dir),
                out_images_dir=str(out_images_dir),
                out_labels_dir=str(out_labels_dir),
                scale=scale
            )

        print(f"\n[Scale x{scale} completed] 输出目录: {out_root}\n")


# =========================
# 8. main
# =========================
if __name__ == "__main__":
    dataset_root = "./datasets/VisDrone"
    output_parent = "./datasets"

    process_dataset(
        dataset_root=dataset_root,
        output_parent=output_parent,
        scales=(0.85,0.75,0.5)
    )
