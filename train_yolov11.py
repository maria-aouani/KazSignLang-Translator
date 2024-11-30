import os
import glob
import torch
from ultralytics import YOLO
from dataclasses import dataclass, field
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split


# Function to load labels
def load_labels(label_dir):
    all_boxes = []
    all_keypoints = []

    for label_file in glob.glob(os.path.join(label_dir, '*.txt')):
        if os.stat(label_file).st_size == 0:
            print(f"Empty label file: {label_file}")
            continue

        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = list(map(float, line.strip().split()))
                if len(parts) < 5 + 21 * 2:
                    print(f"Invalid label format in {label_file}: {line.strip()}")
                    continue

                cls = int(parts[0])
                bbox = parts[1:5]
                keypoints = parts[5:]
                all_boxes.append((cls, bbox))
                all_keypoints.append(keypoints)

    return all_boxes, all_keypoints


# Function to apply data augmentations
def apply_transforms(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(0.1),
        transforms.ToTensor()
    ])
    return transform(image)


# Function to parse label files
def parse_label_file(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"No data found in label file: {label_path}")

    hand1 = list(map(float, lines[0].split()))
    bbox1 = hand1[1:5]
    keypoints1 = hand1[5:]

    bbox2, keypoints2 = (None, None)
    if len(lines) > 1:
        hand2 = list(map(float, lines[1].split()))
        bbox2 = hand2[1:5]
        keypoints2 = hand2[5:]

    return bbox1, keypoints1, bbox2, keypoints2


# Function to plot image with keypoints
def plot_image_with_keypoints(image_path, bbox1, keypoints1, bbox2=None, keypoints2=None):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Image shape is incorrect: {img.shape}")

    img_h, img_w = img.shape[:2]

    # Function to draw a single hand
    def draw_hand(bbox, keypoints):
        cx, cy, w, h = bbox
        x1, y1 = int((cx - w / 2) * img_w), int((cy - h / 2) * img_h)
        x2, y2 = int((cx + w / 2) * img_w), int((cy + h / 2) * img_h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 20),
            (4, 5), (5, 6), (6, 7), (7, 20),
            (8, 9), (9, 10), (10, 11), (11, 20),
            (12, 13), (13, 14), (14, 15), (15, 20),
            (16, 17), (17, 18), (18, 20), (2, 7)
        ]

        for connection in hand_connections:
            kp_start, kp_end = connection[0] * 2, connection[1] * 2
            x_start, y_start = int(keypoints[kp_start] * img_w), int(keypoints[kp_start + 1] * img_h)
            x_end, y_end = int(keypoints[kp_end] * img_w), int(keypoints[kp_end + 1] * img_h)
            cv2.line(img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.circle(img, (x_start, y_start), 5, (0, 0, 255), -1)
            cv2.circle(img, (x_end, y_end), 5, (0, 0, 255), -1)

    draw_hand(bbox1, keypoints1)
    if bbox2 and keypoints2:
        draw_hand(bbox2, keypoints2)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


@dataclass(frozen=True)
class TrainingConfig:
    DATASET_YAML: str = "/media/robotics-300/8bf075d6-3600-4c70-8f2d-0d11f1ca9e25/robotics300/PycharmProjects/MediaPipe_Yolo_Lstm_222/MP_Data/kazakh_sign_language.yaml"
    MODEL: str = "yolo11n-pose.pt"
    EPOCHS: int = 10
    KPT_SHAPE: tuple = (21, 2)
    PROJECT: str = "Hand_Keypoints"
    NAME: str = field(default_factory=lambda: f"yolo11n-pose_10_epochs")
    CLASSES_DICT: dict = field(default_factory=lambda: {0: "Ғ", 1: "Әже"})


@dataclass(frozen=True)
class DatasetConfig:
    IMAGE_SIZE: int = 640
    BATCH_SIZE: int = 16
    CLOSE_MOSAIC: int = 10
    MOSAIC: float = 0.4
    FLIP_LR: float = 0.0


def main():
    train_config = TrainingConfig()
    data_config = DatasetConfig()

    # Check GPU usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model and start training
    model = YOLO(train_config.MODEL)
    model.train(
        data=train_config.DATASET_YAML,
        epochs=train_config.EPOCHS,
        imgsz=data_config.IMAGE_SIZE,
        batch=data_config.BATCH_SIZE,
        device=device,
    )


if __name__ == "__main__":
    main()
