import torchvision
from torchvision import datasets, transforms, models
import numpy as np
import torch 
from torch import nn
from torch import optim
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json
import argparse

def argumentParser():
    #setup parser
    parser = argparse.ArgumentParser(description='Training the network')
    parser.add_argument('data_dir', action="store")
    parser.add_argument('--save_dir', action="store", dest="checkpoint_dir", default='mymodel.pth')
    parser.add_argument('--arch', action="store", dest="architecture", default="densenet121")
    parser.add_argument('--learning_rate', action='store', dest='learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', action='store', dest='epochs', type=int, default=5)
    parser.add_argument('--hidden_units', action='store', dest='hidden_units', type=int, default=500)
    parser.add_argument('--gpu', dest='gpu', action='store_true', default=False)
    return parser.parse_args()

def transformData(data):
    print('Tranforming data...')
    train_dir = data + '/train'
    valid_dir = data + '/valid'
    test_dir = data + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    
    validation_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])
    #Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    test_data = datasets.ImageFolder(test_dir ,transform = test_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform = validation_transforms)
    return train_data, test_data, validation_data
    
def loadData(train_data, test_data, validation_data):
    #Using the image datasets and the trainforms, define the dataloaders
    print('Loading data...')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle = True) 
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size = 64,shuffle = True)
    return train_loader, test_loader, validation_loader

def defineModel(arch, hidden_units):
    print('Defining model:')
    model = getattr(models, arch)(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 500)),
                          ('dropout', nn.Dropout(p=0.6)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    print(model)
    return model

def trainModel(model, epochs, learning_rate, train_loader, test_loader, validation_loader, gpu):
    # Define loss and optimizer
    print_every = 30
    steps = 0
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    if (gpu == True and torch.cuda.is_available()):
        device = 'cuda'
        print('Utilising GPU for training!')        
    else:
        device = 'cpu'
        print("Utilising CPU for training!")
        
    model.to(device)
    
    print("Training begins...")

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
        
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0

                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validation_loader, criterion, device)
                    print("Epoch: {}/{} | ".format(e+1, epochs),
                        "Training Loss: {:.4f} | ".format(running_loss/print_every),
                        "Validation Loss: {:.4f} | ".format(valid_loss/len(test_loader)),
                        "Validation Accuracy: {:.4f}".format(accuracy/len(test_loader)))
            
            running_loss = 0
            model.train()

    print("\nTraining completed!")
    return model, optimizer
    
def validation(model, test_loader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    for ii, (inputs, labels) in enumerate(test_loader):
        
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy
    
def main():
    arguments = argumentParser()
    train_data, test_data, validation_data = transformData(arguments.data_dir)
    train_loader, test_loader, validation_loader = loadData(train_data, test_data, validation_data)
    model= defineModel(arguments.architecture, arguments.hidden_units)
    model, optimizer = trainModel(model, arguments.epochs, arguments.learning_rate, train_loader, test_loader, validation_loader, arguments.gpu)
    
    #save the checkpoint
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'architecture': arguments.architecture,
             'classifier': model.classifier,
             'class_to_idx': model.class_to_idx,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epochs': arguments.epochs}

    torch.save(checkpoint, arguments.checkpoint_dir)
    print('Checkpoint ' + arguments.checkpoint_dir + ' created for the trained model.' )

if __name__ == "__main__":
    main()