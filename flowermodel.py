import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.optim import Adam
import torch.nn as nn
from flowernet import FlowerClassifierCNNModel
from util import show_transformed_image
from torch.utils.data import random_split
from PIL import Image
import torch


class FlowerModel:
    def __init__(self, data_folder="./data/"):
        self.cnn_model = FlowerClassifierCNNModel()
        self.optimizer = Adam(self.cnn_model.parameters())
        self.loss_fn = nn.CrossEntropyLoss()
        # load data

    def pre_processing(self, data_folder='./data/'):
        self.transformations = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.total_dataset = datasets.ImageFolder(data_folder, transform=self.transformations)
        self.dataset_loader = DataLoader(dataset=self.total_dataset, batch_size=100)
        items = iter(self.dataset_loader)
        image, label = items.next()
        show_transformed_image(make_grid(image))

    def train(self, n_epoches=10):
        train_size = int(0.8 * len(self.total_dataset))
        test_size = len(self.total_dataset) - train_size
        train_dataset, self.test_dataset = random_split(self.total_dataset, [train_size, test_size])

        train_dataset_loader = DataLoader(dataset=train_dataset, batch_size=100)
        self.test_dataset_loader = DataLoader(dataset=self.test_dataset, batch_size=100)
        for epoch in range(n_epoches):
            self.cnn_model.train()
            for i, (images, labels) in enumerate(train_dataset_loader):
                self.optimizer.zero_grad()
                # print(labels)
                outputs = self.cnn_model(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

    def predict(self, fileName="./data/dandelion/13920113_f03e867ea7_m.jpg"):
        test_image = Image.open(fileName)
        test_image_tensor = self.transformations(test_image).float()
        test_image_tensor = test_image_tensor.unsqueeze_(0)
        output = self.cnn_model(test_image_tensor)
        class_index = output.data.numpy().argmax()
        test_acc_count = 0
        for k, (test_images, test_labels) in enumerate(self.test_dataset_loader):
            test_outputs = self.cnn_model(test_images)
            _, prediction = torch.max(test_outputs.data, 1)
            test_acc_count += torch.sum(prediction == test_labels.data).item()

        test_accuracy = test_acc_count / len(self.test_dataset)
        print('\n \n \n')
        print('The Predicted Value is: ' + str(class_index))
        if class_index == 0:
            print('This is a Daisy ')
        elif class_index == 1:
            print('This is a Dandelion ')
        elif class_index == 2:
            print('This is a Rose ')
        elif class_index == 3:
            print('This is a Sunflower ')
        elif class_index == 4:
            print('This is a Tulip ')
        else:
            print('Flower cannot be determined ')
        print('The Test Accuracy is: ' + str(test_accuracy))

    def saveModel(self, path='flowerData.pth'):
        checkpoint = {'model': self.cnn_model.state_dict(), 'state_dict': self.cnn_model.state_dict(),
                      'optimizer': self.optimizer.state_dict()}
        torch.save(checkpoint, path)
