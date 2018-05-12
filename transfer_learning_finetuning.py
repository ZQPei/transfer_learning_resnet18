import os
import time
import copy

import numpy as np
import matplotlib as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms


# define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# load data and do data augmention
path = 'data/'
mode = ('train', 'val')
transform = {
    'train':transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
    'val':transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
}
kwargs = {'num_workers':4, 'pin_memory':True}

image_datasets = {x: datasets.ImageFolder(root=os.path.join(path, x), transform = transform[x])
                        for x in mode}

data_loaders = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, **kwargs)
                        for x in mode}
class_names = image_datasets['train'].classes
dataset_size = {x: len(image_datasets[x]) for x in mode}

# define my net and criterion optimizer
my_resnet18 = torchvision.models.resnet18(pretrained=True)
num_features = my_resnet18.fc.in_features
my_resnet18.fc = nn.Linear(512, 2)
#my_resnet18 = nn.DataParallel(my_resnet18)
my_resnet18 = my_resnet18.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(my_resnet18.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

print(my_resnet18)
#import ipdb; ipdb.set_trace()

# train
def train_model(epochs=25):

    best_model_wts = copy.deepcopy(my_resnet18.state_dict())
    best_acc = 0.

    for epoch in range(epochs):
        # in each epoch
        #epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch, epochs-1))
        print('-'*10)
        # iterate on the whole data training set
        for phase in mode:
            running_loss = 0.
            running_corrects = 0
            if phase == 'train':
                exp_lr_scheduler.step()
                my_resnet18.train()
            else :
                my_resnet18.eval()

            # in each epoch iterate over all dataset
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                with torch.set_grad_enabled(phase == 'train'):
                    # in each iter step
                    # 1. zero the parameter gradients
                    optimizer.zero_grad()

                    # 2. forward 
                    outputs = my_resnet18(inputs)

                    # 3. compute loss and backward and update parameters
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # statistics
                preds = outputs.max(1)[1]
                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds == labels)
            
            epoch_loss = running_loss/dataset_size[phase]
            epoch_acc = running_corrects.double()/dataset_size[phase]

            print('%s Loss: %.4f ACC: %.4f'%(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(my_resnet18.state_dict())
        
        print()

    # load best model weights
    my_resnet18.load_state_dict(best_model_wts)

train_model()



