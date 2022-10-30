import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # ==========================
        # TODO 1: build your network
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(in_features=(32 * 64 * 64), out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=10)
        
        self.output = nn.Softmax(dim=1)
        # ==========================


    def forward(self, x):
        # (batch_size, 3, 256, 256)

        # ========================
        # TODO 2: forward the data
        out = self.relu(self.conv1(x))
        # (batch_size, 16, 256, 256)
        out = self.pool(out)
        # (batch_size, 16, 128, 128)
        out = self.relu(self.conv2(out))
        # (batch_size, 32, 128, 128)
        out = self.pool(out)
        # (batch_size, 32, 64, 64)

        out = torch.flatten(out, 1)

        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.output(out)
        # ========================
        return out


def calc_acc(output, target):
    predicted = torch.max(output, 1)[1]
    num_samples = target.size(0)
    num_correct = (predicted == target).sum().item()
    return num_correct / num_samples


def training(model, device, train_loader, criterion, optimizer):
    # ===============================
    # TODO 3: switch the model to training mode
    model.train()
    # ===============================
    train_acc = 0.0
    train_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        #print(data)
        #print(target)
        # =============================================
        # TODO 4: initialize optimizer to zero gradient
        optimizer.zero_grad()
        # =============================================

        output = model(data)
        # =================================================
        # TODO 5: loss -> backpropagation -> update weights
        loss = criterion()
        loss = loss(output,target)
        loss.backward()
        optimizer.step()
        # =================================================
        #print(output)
        train_acc += calc_acc(output, target)
        train_loss += loss.item()

    train_acc /= len(train_loader)
    train_loss /= len(train_loader)

    return train_acc, train_loss


def validation(model, device, valid_loader, criterion):
    # ===============================
    # TODO 6: switch the model to validation mode
    model.eval()
    # ===============================
    valid_acc = 0.0
    valid_loss = 0.0

    # =========================================
    # TODO 7: turn off the gradient calculation
    with torch.no_grad():
    # =========================================
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            # ================================
            # TODO 8: calculate accuracy, loss
            loss = criterion()
            loss = loss(output,target)
            valid_acc += calc_acc(output, target)
            valid_loss += loss.item()
            # ================================

    valid_acc /= len(valid_loader)
    valid_loss /= len(valid_loader)

    return valid_acc, valid_loss


def main():
    torch.cuda.empty_cache()
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    # ==================
    # TODO 9: set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # ==================


    # ========================
    # TODO 10: hyperparameters
    # you can add your parameters here
    LEARNING_RATE = 0.01
    BATCH_SIZE = 64
    EPOCHS = 250
    TRAIN_DATA_PATH = "./data/train/"
    VALID_DATA_PATH = "./data/valid/"
    MODEL_PATH = "model.pt"

    # ========================


    # ===================
    # TODO 11: transforms
    transform_set = [ transforms.RandomRotation(10),  
                      transforms.RandomAffine(0, scale=(0.8, 1.2)),
                      transforms.RandomAffine(0, shear=10)]

    train_transform = transforms.Compose([
        # may be adding some data augmentations?
        transforms.RandomApply(transform_set, p=0.8),
        #torchvision.transforms.RandomResizedCrop(size=224,scale=(0.08, 1.0)),
        #torchvision.transforms.RandomHorizontalFlip(),
        #transforms.Resize(28),
        #transforms.CenterCrop(224),
        #transforms.RandomAffine(degrees=(-30,30)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    train_transform2 = transforms.Compose([
        #transforms.RandomApply(transform_set, p=0.7),
        #torchvision.transforms.RandomResizedCrop(size=224,scale=(0.08, 1.0)),
        #torchvision.transforms.RandomHorizontalFlip(),
        #transforms.Resize(28),
        #transforms.CenterCrop(224),
        transforms.RandomAffine(degrees=(-30,30)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    train_transform3 = transforms.Compose([
        #torchvision.transforms.RandomResizedCrop(size=224,scale=(0.08, 1.0)),
        #torchvision.transforms.RandomHorizontalFlip(),
        #transforms.Resize(28),
        #transforms.CenterCrop(224),
        transforms.RandomAffine(degrees=(-60,60)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    # ===================
    valid_transform = transforms.Compose([
        #torchvision.transforms.Resize(256),
        #torchvision.transforms.CenterCrop(224),
        #transforms.Resize(28),
        #transforms.CenterCrop(224),
        #transforms.RandomAffine(degrees=(-30,30)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


    # =================
    # TODO 12: set up datasets
    # hint: ImageFolder?
    train_data = torchvision.datasets.ImageFolder(TRAIN_DATA_PATH,transform=train_transform)
    train_data += torchvision.datasets.ImageFolder(TRAIN_DATA_PATH,transform=train_transform2)
    #train_data += torchvision.datasets.ImageFolder(TRAIN_DATA_PATH,transform=train_transform3)
    valid_data = torchvision.datasets.ImageFolder(VALID_DATA_PATH,transform=valid_transform)
    # =================


    # ============================
    # TODO 13 : set up dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True)
    # ============================
    '''
    to_pil_image = transforms.ToPILImage()
    for image, label in train_loader:
        img = to_pil_image(image[0])
        img.show()
        
        plt.imshow(img)
        plt.show()
    '''
    
    # build model, criterion and optimizer
    model = Net().to(device).train()
    
    # ================================
    # TODO 14: criterion and optimizer
    criterion = nn.CrossEntropyLoss
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    # ================================
    

    # training and validation
    train_acc = [0.0] * EPOCHS
    train_loss = [0.0] * EPOCHS
    valid_acc = [0.0] * EPOCHS
    valid_loss = [0.0] * EPOCHS

    print('Start training...')
    for epoch in range(EPOCHS):
        print(f'epoch {epoch} start...')

        train_acc[epoch], train_loss[epoch] = training(model, device, train_loader, criterion, optimizer)
        valid_acc[epoch], valid_loss[epoch] = validation(model, device, valid_loader, criterion)

        print(f'epoch={epoch} train_acc={train_acc[epoch]} train_loss={train_loss[epoch]} valid_acc={valid_acc[epoch]} valid_loss={valid_loss[epoch]}')
    print('Training finished')


    # ==================================
    # TODO 15: save the model parameters
    torch.save(model.state_dict(), MODEL_PATH)
    # ==================================


    # ========================================
    # TODO 16: draw accuracy and loss pictures
    # lab2_teamXX_accuracy.png, lab2_teamXX_loss.png
    # hint: plt.plot
    x = np.linspace(0, EPOCHS-1, EPOCHS)
    plt.title("Model Accuracy")
    plt.plot(x,train_acc)
    plt.plot(x,valid_acc)
    plt.plot(train_acc,label="train")
    plt.plot(valid_acc,label="validation")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig("Lab2_Team1_Accuracy.png")
    plt.show()
    
    plt.title("Model Loss")
    plt.plot(x,train_loss)
    plt.plot(x,valid_loss)
    plt.plot(train_loss,label="train")
    plt.plot(valid_loss,label="validation")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig("Lab2_Team1_Loss.png")
    plt.show()

    # =========================================


if __name__ == '__main__':
    main()