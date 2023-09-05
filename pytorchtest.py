import torch
import torch.nn as nn


class conv1(nn.Module):
    def __init__(self):
        super(conv1, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class mlpmixerlayer(nn.Module):
    def __init__(self):
        super(mlpmixerlayer, self).__init__()
        
        self.ln1 = nn.LayerNorm(32)
        self.fc11 = nn.Linear(32, 128)
        self.relu1 = nn.ReLU()
        self.fc12 = nn.Linear(128, 32)

        self.ln2 = nn.LayerNorm(32)
        self.fc21 = nn.Linear(32, 128)
        self.relu2 = nn.ReLU()
        self.fc22 = nn.Linear(128, 32)

    def forward(self, x):
        x1 = self.ln1(x)
        x1 = x1.permute(0, 2, 1)
        x1 = self.fc11(x)
        x1 = self.relu1(x)
        x1 = self.fc12(x)
        x1 = x1.permute(0, 2, 1)
        x = x + x1

        x1 = self.ln2(x)
        x1 = self.fc21(x1)
        x1 = self.relu2(x1)
        x1 = self.fc22(x1)

        return x + x1


class mlpmixer(nn.Module):
    def __init__(self):
        super(mlpmixer, self).__init__()
        
        self.fc0 = nn.Linear(32, 32)
        self.l1 = mlpmixerlayer()
        self.l2 = mlpmixerlayer()
        self.l3 = mlpmixerlayer()


    def forward(self, x):
        x = self.fc0(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        return x


# Tworzenie instancji modelu
model1 = conv1()
model2 = mlpmixer()
print(model1)
print()
print(model2)
print()

# Funkcja do zliczania parametrów
def licz_parametry(model):
    parametry = 0
    for parametr in model.parameters():
        if parametr.requires_grad:
            parametry += parametr.numel()
    return parametry

lparam1 = licz_parametry(model1)
lparam2 = licz_parametry(model2)
print(f'Liczba parametrów w modelu 1: {lparam1}')
print(f'Liczba parametrów w modelu 2: {lparam2}')
