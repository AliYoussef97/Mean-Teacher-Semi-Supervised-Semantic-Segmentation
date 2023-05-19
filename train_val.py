import torch
import torch.optim as optim
from mt_functions import sigmoid_rampup, update_ema
from loss_functions import sigmoid_mse_loss, BCELogitsLoss, ValBCELogitsLoss
from utils import accuracy_fn, dice_acc, BJI_acc
from tqdm import tqdm
import numpy as np
import time
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_val(student, teacher, train_loader, validation_loader, device, args):

    """
    train function:

        - This function is where the code mostly runs. It takes the student and teacher
          models, training and valdiation dataloaders, device and args as arguments.

        - The Optimizer chosen was Adam, otpimzing the student model with a lr of 0.0001.

        - alpha smoothing coefficient was 0.999 with rampup length of 12 and consitency hyperparameter as 4.
        
        - clssification loss was BCEWithLogitsLoss and consistency loss was calculated using sigmoid_mse_loss function.
         
        - Every epoch, the lambda ramup increases, until epoch 12 where it becomes constant till the end.

        - The train_loader contains a mixed of labaled and unlabaled images and labaels. img1 and img2 are the
          same image with different noise, the label is binary or -1 if unlabaled.
        
        - The student prediction is computed using img1 then the classification loss between it and the label.
          The teacher prediction is computed using img2 with no gradient, the consistency loss is computed 
          between the student prediction and teacher prediction, then multiplied by the consitency weight.

        - The losses are added up, a backward propagation is done and the weights of the students are updated,
          then the weights of the teacher are updated using the update_ema function.

        - A Validation loop is created that loops over the validation loader, the model is evaluated based on
          validation accuracy, element-wise pixel accuracy, dice score and Binary Jaccard Index.
          

    Inputs:
        - student: The student's model, type: pytorch model

        - mean_teacher: The teacher's model, type: pytorch model

        - train_loader: Training data loader containing laballed/unlaballed noisy images and laballed/unlaballed masks.

        - validation_loader: Validation data loader containing images and labels.

        - device: cuda or cpu.

        - args: arguments using in training function.

    Outputs:
        - None (The models are saved at the end of training)
    """

    print("TRAINING.....")

    optimizer = optim.Adam(student.parameters(),args.lr) 
    scheduler = CosineAnnealingLR(optimizer,T_max=args.lr_ramp_down,last_epoch=-1,eta_min=0)
    
    logs = np.zeros((args.epochs, 8))

    for epoch in range(args.epochs):
        
        time_start = time.time()
        running_loss = 0
        labaled_iter = tqdm(train_loader)
        
        if epoch <= args.consistency_rampup:
            lambda_ramp = sigmoid_rampup(epoch,args.consistency_rampup)
               
        
        for train_data in labaled_iter:
                        
            img1, img2, label = train_data
            
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            mini_batch_size = img1.size(0)
            
            student_pred = student(img1)["out"] 
            
            class_loss = BCELogitsLoss(student_pred,label)/mini_batch_size

            with torch.no_grad():
                teacher_pred = teacher(img2)["out"]
             
            consistency_weight = args.consistency*lambda_ramp

            consis_loss = consistency_weight*sigmoid_mse_loss(student_pred,teacher_pred)/mini_batch_size
            
            loss = class_loss + consis_loss

            running_loss += loss.item()*mini_batch_size
            
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            args.global_step+=1

            update_ema(student,teacher,args.alpha,args.global_step)

            labaled_iter.set_description(f'learning_rate = {optimizer.param_groups[0]["lr"]}, Consistency weight: {consistency_weight: .3f} , Class loss: {class_loss: .6f}, Consis loss: {consis_loss: .6f}')
        
        running_loss /= len(train_loader)
        
        stop_time = time.time()
        epoch_time = stop_time-time_start
        
        
        student.eval()
        with torch.no_grad():
            
            val_running_loss = 0
            total_accuracy = 0
            dice_score = 0
            BJI_score = 0
            
            for val_data in validation_loader:
                
                val_imgs , val_labels = val_data

                val_imgs, val_labels = val_imgs.to(device), val_labels.to(device)
                
                val_mini_batch_size = val_imgs.size(0)

                val_preds = student(val_imgs)["out"]
                
                val_loss = ValBCELogitsLoss(val_preds,val_labels)/val_mini_batch_size
                
                val_running_loss += val_loss.item()*val_mini_batch_size
            
                total_accuracy += accuracy_fn(val_preds,val_labels)
    
                dice_score += dice_acc(val_preds,val_labels,device)
        
                BJI_score += BJI_acc(val_preds,val_labels,device)
        
        student.train()

        val_running_loss /= len(validation_loader)

        avg_accuracy = total_accuracy/len(validation_loader)

        avg_dice = dice_score/len(validation_loader)*100

        avg_BJI = BJI_score/len(validation_loader)*100

        print(f"Epoch: {epoch+1}, Running Train loss: {running_loss:.6f}, Running Validation loss: {val_running_loss:.6f}, Validation Accuracy: {avg_accuracy:.4f}, Dice Score: {avg_dice:.4f}, BJI Score: {avg_BJI:.4f}")

        
        logs[epoch] = np.array([epoch, epoch_time, optimizer.param_groups[0]['lr'], running_loss, val_running_loss, avg_accuracy, avg_dice, avg_BJI])
        np.savetxt('logs.csv', logs, delimiter=',')
        
        
        scheduler.step()
        
        if epoch % 10 == 0:
            torch.save(student.state_dict(), 'student_semi_supervised.pt')
            torch.save(teacher.state_dict(), 'teacher_semi_supervised.pt')