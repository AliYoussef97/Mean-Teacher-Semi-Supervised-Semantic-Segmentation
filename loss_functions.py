import torch.nn as nn
from segmentation_models_pytorch.losses import SoftBCEWithLogitsLoss


def sigmoid_mse_loss(input_pred, psuedo_label):
    """
    sigmoid_mse_loss function:
        - This function applies sigmoid on both the prediction and
          the pseudo label then computes the MSE lsos between them.

    Inputs:
        - input_pred: The model's prediction, type:torch.float32

        - pseudo_label: The pseudo label, type: torch.float32
    
    Outputs:
        - Sigmoid MSE Loss, type: float
    """

    mse = nn.MSELoss()
    sigmoid = nn.Sigmoid()
    input_pred = sigmoid(input_pred)
    psuedo_label = sigmoid(psuedo_label)
    num_classes = input_pred.size(1)

    return mse(input_pred, psuedo_label) / num_classes


def BCELogitsLoss(student_prediction,label):
    """
    BCELogitsLoss function:
        - This function computes the Binary Cross-Entropy Loss between
          the student's prediction and the ground truth label while
          unlabelled labels are ignored.

    Inputs:
        - input_pred: The model's prediction, type:torch.float32

        - label: The ground truth label, type: torch.float32
    
    Outputs:
        - Binary Cross-Entropy Loss, type: float
    """

    label = label.float()
    train_soft_bce = SoftBCEWithLogitsLoss(ignore_index=-1)
    classsification_loss = train_soft_bce(student_prediction,label)

    return classsification_loss


def ValBCELogitsLoss(student_prediction,label):
    """
    BCELogitsLoss function:
        - This function computes the Binary Cross-Entropy Loss between
          the student's prediction and the ground truth label during 
          validation.

    Inputs:
        - input_pred: The model's prediction, type:torch.float32

        - label: The ground truth label, type: torch.float32
    
    Outputs:
        - Binary Cross-Entropy Loss, type: float
    """

    label = label.float()
    val_soft_bce = SoftBCEWithLogitsLoss()
    val_classsification_loss = val_soft_bce(student_prediction,label)

    return val_classsification_loss