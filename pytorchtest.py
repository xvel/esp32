import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Pytorch version: {torch.__version__}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("No GPU available.")


class mlp(nn.Module):
    def __init__(self):
        super(mlp, self).__init__()
        
        self.rgb_conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(1024, 48)
        self.relu = nn.ReLU()
        self.fcout = nn.Linear(48, 10)

    def forward(self, x):
        x = self.rgb_conv(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        x = self.relu(x)
        x = self.fcout(x)

        return x


class conv1(nn.Module):
    def __init__(self):
        super(conv1, self).__init__()
        
        self.rgb_conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu = nn.Hardswish()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=48, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(48 * 8 * 8, 10)

    def forward(self, x):
        x = self.rgb_conv(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class mlpmixerlayer(nn.Module):
    def __init__(self, dim):
        super(mlpmixerlayer, self).__init__()
        
        self.ln1 = nn.LayerNorm(32)
        self.fc11 = nn.Linear(32, dim)
        self.fc12 = nn.Linear(dim, 32)

        self.ln2 = nn.LayerNorm(32)
        self.fc21 = nn.Linear(32, dim)
        self.fc22 = nn.Linear(dim, 32)

        self.hardswish = nn.Hardswish()

    def forward(self, x):
        x1 = self.ln1(x)
        x1 = x1.permute(0, 2, 1)
        x1 = self.fc11(x1)
        x1 = self.hardswish(x1)
        x1 = self.fc12(x1)
        x1 = x1.permute(0, 2, 1)
        x = x + x1

        x1 = self.ln2(x)
        x1 = self.fc21(x1)
        x1 = self.hardswish(x1)
        x1 = self.fc22(x1)

        return x + x1


class mlpmixer(nn.Module):
    def __init__(self, num_layers, repeat, dim, dim2):
        super(mlpmixer, self).__init__()
        self.repeat = repeat
        
        self.rgb_conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.fc0 = nn.Linear(32, 32)
        self.layers = nn.ModuleList([mlpmixerlayer(dim) for _ in range(num_layers)])
        
        self.fc1 = nn.Linear(32, dim2)
        self.hardswish = nn.Hardswish()
        self.avg = nn.AvgPool1d(32)
        self.fcout = nn.Linear(dim2, 10)

    def forward(self, x):
        x = self.rgb_conv(x)
        x = x.squeeze(dim=1)

        x = self.fc0(x)
        
        for layer in self.layers:
            for _ in range(self.repeat):
                x = layer(x)

        x = self.fc1(x)
        x = self.hardswish(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fcout(x)

        return x


class mlpmixerlayer2(nn.Module):
    def __init__(self, dim):
        super(mlpmixerlayer2, self).__init__()
        
        self.ln1 = nn.LayerNorm(32)
        self.fc1 = nn.Linear(32, dim)
        self.fc2 = nn.Linear(dim, 32)

        self.hardswish = nn.Hardswish()

    def forward(self, x):
        x1 = self.ln1(x)

        x2 = torch.permute(x1, (0, 2, 1))
        x2 = self.fc1(x2)
        x2 = self.hardswish(x2)
        x2 = self.fc2(x2)
        x2 = torch.permute(x2, (0, 2, 1))

        x1 = self.fc1(x1)
        x1 = self.hardswish(x1)
        x1 = self.fc2(x1)

        return x + x1 + x2


class mlpmixer2(nn.Module):
    def __init__(self, num_layers, repeat, dim, dim2):
        super(mlpmixer2, self).__init__()
        self.repeat = repeat
        
        self.rgb_conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.fc0 = nn.Linear(32, 32)
        self.layers = nn.ModuleList([mlpmixerlayer2(dim) for _ in range(num_layers)])
        
        self.fc1 = nn.Linear(32, dim2)
        self.hardswish = nn.Hardswish()
        self.avg = nn.AvgPool1d(32)
        self.fcout = nn.Linear(dim2, 10)

    def forward(self, x):
        x = self.rgb_conv(x)
        x = x.squeeze(dim=1)

        x = self.fc0(x)
        
        for layer in self.layers:
            for _ in range(self.repeat):
                x = layer(x)

        x = self.fc1(x)
        x = self.hardswish(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fcout(x)

        return x


def train_model(model, trainloader, optimizer, scheduler, criterion, num_epochs):
    model.train()
    loss_values = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
        loss_values.append(running_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss:.4f}')

    print("Training completed.")

    return loss_values


def evaluate_model(model, testloader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on test images: {accuracy}%')

    return accuracy


# Hyperparameters
batch_size = 64
learning_rate = 0.01
num_epochs = 30

# Transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Initialize models and optimizers
#mod1 = torch.compile(mlpmixer(1, 1, 128, 256))
#mod2 = torch.compile(mlpmixer(1, 1, 128, 256))
#mod3 = torch.compile(mlpmixer(1, 1, 128, 512))
#mod4 = torch.compile(mlpmixer(1, 1, 128, 1024))

models = [mlpmixer2(2, 1, 256, 256)]

optimizers = [optim.Adam(model.parameters(), lr=learning_rate) for model in models]

# OneCycleLR scheduler
schedulers = [OneCycleLR(optimizer, max_lr=learning_rate, epochs=num_epochs, steps_per_epoch=len(trainloader)) for optimizer in optimizers]

# Lists to store loss values for plotting
loss_values = [[] for _ in models]

# Train models
for i, model in enumerate(models):
    loss_values[i] = train_model(model, trainloader, optimizers[i], schedulers[i], nn.CrossEntropyLoss(), num_epochs)

# Evaluate models
accuracies = [evaluate_model(model, testloader) for model in models]

# Plot loss values
plt.figure(figsize=(10, 5))
for i, loss in enumerate(loss_values):
    plt.plot(range(1, num_epochs + 1), loss, label=f'Model {i+1}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)
plt.show()

# Print accuracies
for i, accuracy in enumerate(accuracies):
    print(f'Accuracy of Model {i+1} on test images: {accuracy}%')

print("dupa")
