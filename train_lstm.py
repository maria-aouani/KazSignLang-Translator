import os
import glob
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, field

# Функция для загрузки меток
def load_labels(label_dir):
    all_boxes = []
    all_keypoints = []

    for label_file in glob.glob(os.path.join(label_dir, '*.txt')):
        if os.stat(label_file).st_size == 0:
            print(f"Пустой файл метки: {label_file}")
            continue

        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = list(map(float, line.strip().split()))
                if len(parts) < 5 + 21 * 2:
                    print(f"Неверный формат метки в {label_file}: {line.strip()}")
                    continue

                cls = int(parts[0])
                bbox = parts[1:5]
                keypoints = parts[5:]
                all_boxes.append((cls, bbox))
                all_keypoints.append(keypoints)

    return all_boxes, all_keypoints

# Преобразования изображений
def apply_transforms(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(0.1),
        transforms.ToTensor()
    ])
    image = transform(image)
    return image

# Функция для извлечения меток из файла
def parse_label_file(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()

    if len(lines) == 0:
        raise ValueError(f"Нет данных в файле метки: {label_path}")

    # Парсим первую руку
    hand1 = list(map(float, lines[0].split()))
    bbox1 = hand1[1:5]
    keypoints1 = hand1[5:]

    if len(lines) > 1:  # Проверяем, есть ли вторая рука
        hand2 = list(map(float, lines[1].split()))
        bbox2 = hand2[1:5]
        keypoints2 = hand2[5:]
    else:
        # Если второй руки нет, ставим значения по умолчанию
        bbox2, keypoints2 = None, None

    return bbox1, keypoints1, bbox2, keypoints2

# Обновленная функция для отображения изображения с ключевыми точками
def plot_image_with_keypoints(image_path, bbox1, keypoints1, bbox2=None, keypoints2=None):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Неверная форма изображения: {img.shape}")

    img_h, img_w = img.shape[:2]

    # Функция для рисования одной руки (bbox и ключевые точки)
    def draw_hand(bbox, keypoints):
        cx, cy, w, h = bbox
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 20),
            (4, 5), (5, 6), (6, 7), (7, 20),
            (8, 9), (9, 10), (10, 11), (11, 20),
            (12, 13), (13, 14), (14, 15), (15, 20),
            (16, 17), (17, 18), (18, 20), (2, 7)
        ]

        for connection in hand_connections:
            kp_start = connection[0] * 2
            kp_end = connection[1] * 2
            x_start = int(keypoints[kp_start] * img_w)
            y_start = int(keypoints[kp_start + 1] * img_h)
            x_end = int(keypoints[kp_end] * img_w)
            y_end = int(keypoints[kp_end + 1] * img_h)
            cv2.line(img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.circle(img, (x_start, y_start), 5, (0, 0, 255), -1)
            cv2.circle(img, (x_end, y_end), 5, (0, 0, 255), -1)

    # Рисуем первую руку
    draw_hand(bbox1, keypoints1)

    # Если есть вторая рука, рисуем ее
    if bbox2 is not None and keypoints2 is not None:
        draw_hand(bbox2, keypoints2)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Класс конфигурации модели BiLSTM
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

# Конфигурация для обучения
@dataclass(frozen=True)
class TrainingConfig:
    DATASET_YAML: str = '/media/robotics-300/8bf075d6-3600-4c70-8f2d-0d11f1ca9e25/robotics300/PycharmProjects/MediaPipe_Yolo_Lstm_222/MP_Data/kazakh_sign_language.yaml'
    EPOCHS: int = 50
    INPUT_SIZE: int = 42  # 21 ключевых точек * 2 для (x, y)
    HIDDEN_SIZE: int = 128
    OUTPUT_SIZE: int = 2  # Количество классов или признаков
    PROJECT: str = "Hand_Keypoints"
    NAME: str = field(default_factory=lambda: f"BiLSTM_50_epochs")

train_config = TrainingConfig()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Пример использования BiLSTM модели
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Пример данных
# Предположим, что у нас есть данные X_train (размерность [batch_size, sequence_length, input_size]) и y_train (размерность [batch_size, output_size])
# В реальном проекте вы будете загружать эти данные из вашего набора данных.
X_train = torch.rand(100, 30, 42)  # Пример 100 последовательностей, каждая длиной 30, с 42 признаками
y_train = torch.randint(0, 2, (100, 2))  # Пример 100 меток для 2 классов

# Преобразуем в DataLoader для мини-батчей
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Инициализация модели
model = BiLSTMModel(input_size=42, hidden_size=128, output_size=2).to(device)

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()  # Используется для многоклассовой классификации
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Процесс обучения
num_epochs = 50
for epoch in range(num_epochs):
    model.train()  # Устанавливаем модель в режим обучения
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Обнуление градиентов
        optimizer.zero_grad()

        # Прямой проход (forward pass)
        outputs = model(inputs)

        # Вычисление потерь
        loss = criterion(outputs, labels)

        # Обратное распространение (backward pass) и оптимизация
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Оценка точности
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Средний убыток и точность
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total

    print(f"Эпоха [{epoch + 1}/{num_epochs}], Потери: {avg_loss:.4f}, Точность: {accuracy:.2f}%")

# Сохранение модели
torch.save(model.state_dict(), 'biLSTM_model.pth')
