# Импорт необходимых библиотек
import os  # Для работы с операционной системой
import random  # Для генерации случайных чисел
import numpy as np  # Для работы с массивами и математическими функциями
import pytorch_lightning as pl  # Для упрощения процесса обучения с PyTorch
import torch  # Основная библиотека для работы с глубоким обучением
import torchvision.transforms as transforms  # Для преобразования изображений
import wandb  # Для логирования экспериментов с Weights & Biases
from pytorch_lightning.loggers import WandbLogger  # Для логирования экспериментов с Weights & Biases
from torch import nn  # Для создания нейронных сетей
from torch.nn import functional as F  # Функции активации и другие функции для нейронных сетей
from torch.utils.data import DataLoader  # Для загрузки данных в батчах
from torchvision.datasets import ImageFolder
0 # Для загрузки изображений из папок


# Функция для фиксации случайных сидов для обеспечения воспроизводимости
def seed_everything(seed):
    random.seed(seed)  # Устанавливаем сид для генератора случайных чисел Python
    os.environ["PYTHONHASHSEED"] = str(seed)  # Устанавливаем сид для хеширования
    np.random.seed(seed)  # Устанавливаем сид для NumPy
    torch.manual_seed(seed)  # Устанавливаем сид для PyTorch (CPU)
    torch.cuda.manual_seed(seed)  # Устанавливаем сид для PyTorch (GPU)
    torch.backends.cudnn.deterministic = True  # Делаем вычисления детерминированными


# Фиксируем сид
seed_everything(123456)

# Определение аугментаций для тренировочного и валидационного наборов данных
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Случайное горизонтальное отражение
    transforms.RandomRotation(10),  # Случайный поворот на 10 градусов
    transforms.ToTensor(),  # Преобразование изображения в тензор
    transforms.Normalize((0.5,), (0.5,))  # Нормализация тензора: (среднее, стандартное отклонение)
])

val_transform = transforms.Compose([
    transforms.ToTensor(),  # Преобразование изображения в тензор
    transforms.Normalize((0.5,), (0.5,))  # Нормализация тензора
])

# Загрузка тренировочного и валидационного наборов данных
train_dataset = ImageFolder('./dataset/train', transform=train_transform)  # Загрузка тренировочного набора
val_dataset = ImageFolder('./dataset/val', transform=val_transform)  # Загрузка валидационного набора


train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=3, persistent_workers=True)  # Загрузчик для тренировочных данных
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=3, persistent_workers=True)  # Загрузчик для валидационных данных

# Определение модели нейронной сети
class YourNet(nn.Module):
    def __init__(self):
        super().__init__()  # Инициализация родительского класса
        # Определение сверточных слоев
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Первый сверточный слой
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Второй сверточный слой
        self.fc1 = nn.Linear(64 * 32 * 32, 256)  # Полносвязный слой
        self.fc2 = nn.Linear(256, 100)  # Полносвязный слой для 100 классов

    def forward(self, x):
        # Прямой проход через сеть
        x = F.relu(self.conv1(x))  # Применение первого сверточного слоя и ReLU
        x = F.max_pool2d(x, kernel_size=2)  # Пуллинг
        x = F.relu(self.conv2(x))  # Применение второго сверточного слоя и ReLU
        x = F.max_pool2d(x, kernel_size=2)  # Пуллинг
        x = x.view(x.size(0), -1)  # Преобразование тензора в вектор (Flatten)
        x = F.relu(self.fc1(x))  # Применение первого полносвязного слоя и ReLU
        x = self.fc2(x)  # Применение второго полносвязного слоя
        return x  # Возврат предсказаний


# Определение класса для тренировки модели
class YourModule(pl.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()  # Инициализация родительского класса
        self.model = model  # Сохранение модели
        self.learning_rate = learning_rate  # Сохранение скорости обучения
        self.loss_fn = nn.CrossEntropyLoss()  # Определение функции потерь

    def forward(self, x):
        return self.model(x)  # Прямой проход через модель

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)  # Оптимизатор Adam

    def training_step(self, train_batch, batch_idx):
        images, labels = train_batch  # Извлечение изображений и меток из батча
        logits = self(images)  # Получение логитов от модели
        loss = self.loss_fn(logits, labels)  # Вычисление потерь
        return loss  # Возврат потерь

    def validation_step(self, val_batch, batch_idx):
        images, labels = val_batch  # Извлечение изображений и меток из валидационного батча
        logits = self(images)  # Получение логитов от модели
        loss = self.loss_fn(logits, labels)  # Вычисление потерь
        acc = (logits.argmax(dim=1) == labels).float().mean()  # Вычисление точности
        self.log('val_loss', loss)  # Логирование потерь на валидации
        self.log('val_acc', acc)  # Логирование точности на валидации


# Запуск тренировки
if __name__ == '__main__':
    # Замените 'your_api_key' на ваш фактический API-ключ W&B
    wandb.login(key='your_api_key')
    wandb_logger = WandbLogger(log_model='all')  # Инициализация логгера Weights & Biases
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Определение устройства (GPU или CPU)

    model = YourNet().to(device)  # Инициализация модели и перенос на устройство
    module = YourModule(model, learning_rate=0.001)  # Инициализация модуля для тренировки

    trainer = pl.Trainer(logger=wandb_logger, max_epochs=10)  # Инициализация тренера
    trainer.fit(module, train_dataloader, val_dataloader)  # Запуск тренировки

    wandb.finish()  # Завершение логирования в Weights & Biases
