import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
import numpy as np
from torchmetrics.classification import BinaryJaccardIndex, Dice

def accuracy_fn(preds,labels):
    """
    accuracy function:
        - This function computes the the element wise accuracy,
        - it is used to return the accuracy of the model's prediction.

    Inputs:
        - y_pred: The model's prediction, type:torch.float32
        - y_true: The grount truth labels, type:torch.float32
    Outputs:
        - acc: The element wise accuracy, type:torch.float32
    """
    sigmoid = nn.Sigmoid()

    preds = sigmoid(preds)
                
    preds = (preds>0.5).float()
    correct = torch.eq(preds, labels).sum().item()
    acc = (correct/torch.numel(labels)) * 100

    return acc


def dice_acc(preds,labels,device):
    """
    dice function:
        - This function computes the dice accuracy,
        - it is used to return the dice accuracy of the model's prediction.

    Inputs:
        - y_pred: The model's prediction, type:torch.float32
        - y_true: The grount truth labels, type:torch.float32
    Outputs:
        - acc: The dice accuracy, type:torch.float32
    """
    dice = Dice().to(device)
    
    dice_score = dice(preds,labels.int()).to(device)
    
    return  dice_score.item()
    

def BJI_acc(preds,labels,device):
    """
    BinaryJacardIndex function:
        - This function computes the BJI score,
        - it is used to return the dice score of the model's prediction.

    Inputs:
        - y_pred: The model's prediction, type:torch.float32
        - y_true: The grount truth labels, type:torch.float32
    Outputs:
        - acc: The BJI accuracy, type:torch.float32
    """
    
    bji = BinaryJaccardIndex().to(device)
    
    BJI_score = bji(preds,labels)
    
    return BJI_score.item()


def plot(img,label,pred):
    """
    plot function:
        - This function plot a test images, ground truth label,
          and the predicted label

    Inputs:
        - img: Validation Images type:torch.float32

        - label: The ground truth labels, type:torch.float32

        - preds: The predicted labels, type:torch.float32

    Outputs:
        - plt.show():  plot a random valdiation image, ground truth label,
          and a predicted label   
    """
    
    sigmoid = nn.Sigmoid()
    pred = sigmoid(pred)            
    pred = (pred>0.5).float()
    r = np.random.randint(img.size(0))
    img = img[r].cpu().numpy().transpose((1,2,0))
    label = label[r].cpu().numpy().squeeze()  
    pred = pred[r].cpu().numpy().squeeze() 
    
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 3, 1)
    plt.title('Image')
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    plt.title('Ground Truth label')
    plt.imshow(label)
    plt.subplot(1, 3, 3)
    plt.title('Student prediction')
    plt.imshow(pred)
    plt.show()