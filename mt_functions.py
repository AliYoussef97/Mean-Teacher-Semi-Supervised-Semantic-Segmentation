import numpy as np
from torchvision import models

def sigmoid_rampup(current, rampup_length):
    """
    sigmoid_rampup function:
        - This function returns a number that is increasing gradually,
          between 0 and 1, depending on current epoch and rampup length.
    
    Inputs:
        - current: current epoch, type: int

        - rampup_length: Consistency ramp-up length
    
    Outputs:
        - Float value: a gradually increasing value between 0 and 1, type: Float

    """

    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
    

def update_ema(student, mean_teacher, alpha, global_step):
    """
    update_ema function:
        - The exponential moving average function updates the weights,
          of the mean_teacher model manually. Using the student's weights,
          alpha, and the global/training step.
    
    Inputs:
        - student: The student's model, type: pytorch model

        - mean_teacher: The teacher's model, type: pytorch model

        - alpha: smoothing coefficient, type: float

        - global_step: the current training step, type: int
    
    Outputs:
        - None

    """

    alpha = min(1 - 1 / (global_step + 1), alpha)

    for mean_teacher_param, student_param in zip(mean_teacher.parameters(), student.parameters()):
        mean_teacher_param.data.mul_(alpha).add_(student_param.data, alpha = 1 - alpha)


def student_teacher_models(device):
    """
    student_teacher_models function:
        - Returns the student model and the teacher model.
    
    Inputs:
        - device: cuda or cpu.
    
    Outputs:
        - student model.

        - teacher model.

    """
    backbone = models.ResNet50_Weights.IMAGENET1K_V2
    student = models.segmentation.deeplabv3_resnet50(weights=None, progress=True, num_classes = 1, weights_backbone = backbone).to(device)
    teacher = models.segmentation.deeplabv3_resnet50(weights=None, progress=True, num_classes = 1, weights_backbone = backbone).to(device)
  
    for param in teacher.parameters():
        param.detach_()
    
    return student, teacher