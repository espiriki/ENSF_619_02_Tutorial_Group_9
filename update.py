#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

#### TUTORIAL FOR ENSF 619.3 ---- Group 9###

## This is the algorithm of FedAvg!!

###---------------------------------------------------###
## Summary of FedAvg:

### In FedAvg, each client uses it own dataset to obtain the gradient updates
### The main server receives these updates from all the clients and then perform averaging of the gradient updates
### The averaged gradients update is then sent back to each client and their weights are updated.






### import python libraries
### this code use pytorch to implement the algo


import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset



### This is class definitiion of pytorch dataset.
### Pytorch allows us to define custom dataset inheriting from Dataset class

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


    

 ### class for updating local clients
class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, device_id):
        self.args = args
        ## split the dataset in to train, validatation and test
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        
        ## set the device to cuda if gpu is available
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        
        
        self.criterion = nn.NLLLoss().to(self.device)
        
        ## store gpu id
        self.device_id = device_id

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        ## this splits the dataset in to train, validation and test in ratio of 80:10:10.
        
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        
        ## create dataloader using pytorch inbuilt function 'DataLoader'
        
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        
        ## pytorch provides two modes eval() and train().
        ## train sets the mode to training and eval is used for inferernce.
        
        model.train()
        epoch_loss = []
        
        # Set optimizer for the local updates
        
        ### set optimizer
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                
                ## push data to the gpu
                images, labels = images.to(self.device), labels.to(self.device)
                
                ## set the gradients to zero before each iteration
                optimizer.zero_grad()
                
                ## obtain the model output
                log_probs = model(images)
                
                ### get the loss for the bach
                loss = self.criterion(log_probs, labels)
                
                ## do backpropagation
                loss.backward()
                
                ### get the gradient updates
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} on device {} | Images on device {} [{:03d}/{:03d} ({:.0f}%)] Loss: {:.3f}'.format(
                        global_round, iter, self.device_id, self.device_id, batch_idx *
                        len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        ## return the model state and loss
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """
        ### we will use the eval mode for inference
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for _, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss
