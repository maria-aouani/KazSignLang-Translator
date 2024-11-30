import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os


# Определение модели BiLSTM
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM слой (Bidirectional)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Полносвязный слой для классификации или предсказания
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 для bidirectional LSTM

    def forward(self, x):
        # Проход через LSTM
        out, _ = self.lstm(x)

        # Берем последний временной шаг
        out = out[:, -1, :]

        # Проход через полносвязный слой
        out = self.fc(out)
        return out


# Пример Dataset для временных рядов
class ExampleDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Функция тренировки
def train(model, criterion, optimizer, train_loader, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Обнуление градиентов
            optimizer.zero_grad()

            # Прямой проход
            outputs = model(inputs)

            # Вычисление потерь
            loss = criterion(outputs, labels)

            # Обратный проход
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


# Функция сохранения модели и оптимизатора
def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


# Функция загрузки модели и оптимизатора
def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {checkpoint_path}. Last epoch: {epoch}, loss: {loss}")
    return epoch, loss


# Пример данных для тренировки
input_size = 10  # Количество признаков в одном временном шаге
hidden_size = 64  # Количество скрытых нейронов
output_size = 2  # Количество классов (например, бинарная классификация)
num_epochs = 20  # Количество эпох
batch_size = 16  # Размер батча

# Пример данных для тренировочного набора
data = np.random.randn(1000, 30, input_size).astype(np.float32)  # 1000 примеров, 30 временных шагов
labels = np.random.randint(0, 2, size=(1000,)).astype(np.long)  # Бинарные метки

# Преобразуем данные в тензоры PyTorch
train_data = torch.tensor(data)
train_labels = torch.tensor(labels)

# Создаем DataLoader
train_dataset = ExampleDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Инициализируем модель, критерий и оптимизатор
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BiLSTMModel(input_size, hidden_size, output_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Путь для сохранения контрольных точек
checkpoint_path = 'bilstm_checkpoint.pth'

# Проверка на наличие сохраненной контрольной точки и продолжение тренировки
if os.path.exists(checkpoint_path):
    start_epoch, _ = load_checkpoint(checkpoint_path, model, optimizer)
else:
    start_epoch = 0

# Тренировка модели с возможностью остановки и продолжения
train(model, criterion, optimizer, train_loader, device, num_epochs=num_epochs)

# Сохраняем контрольную точку после окончания тренировки
save_checkpoint(model, optimizer, num_epochs, loss.item(), checkpoint_path)
