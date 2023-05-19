from torchvision import transforms
import torch
from PIL import Image
from utils import plot, accuracy_fn, dice_acc, BJI_acc
import os
from data import Data
import random


def test(test_data, student, device, args):
    """
    test function:
        - This function loads the trained model, and plots x number
          of ground truth images, labels and predicted labels from
          the test data. The element-wise pixel accuracy, Dice and
          Binary Jaccard Index scores are also computed.

    Inputs:
        - test_data: A list of image names from the test data.
        
        - student: The student's model, type: pytorch model

        - device: cuda or cpu.

        - args: arguments using in training function.

    Outputs:
        - None   
    """

    print("TESTING.....")


    student.load_state_dict(torch.load('.\student_semi_supervised.pt', map_location=torch.device(device)))
    
    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    student.eval()

    sampled_test_images = random.sample(test_data,20)
    
    test_accuracy = 0
    
    test_dice = 0
    
    test_BJI = 0

    for i in sampled_test_images:

        image  = os.path.join(args.img_path, i[0] + ".jpg")
        trimap = os.path.join(args.seg_path, i[0] + ".png")

        RGB_image = Image.open(image).convert("RGB")
        trimap_image = Image.open(trimap)

        test_image = norm(to_tensor(RGB_image)).to(device)
        test_trimap = to_tensor(trimap_image).to(device)
        test_trimap = Data.mask_blend(test_trimap)

        test_image, test_trimap = test_image.unsqueeze(0), test_trimap.unsqueeze(0)
        
        test_pred = student(test_image)["out"]

        plot(test_image, test_trimap, test_pred)
        
        test_accuracy += accuracy_fn(test_pred, test_trimap)
    
        test_dice += dice_acc(test_pred, test_trimap, device)
        
        test_BJI += BJI_acc(test_pred, test_trimap, device)

    avg__test_accuracy = test_accuracy/len(sampled_test_images)

    avg_test_dice = test_dice/len(sampled_test_images)*100

    avg_test_BJI = test_BJI/len(sampled_test_images)*100

    print(f"Test Accuracy: {avg__test_accuracy:.4f}, Test Dice Score: {avg_test_dice:.4f}, Test BJI Score: {avg_test_BJI:.4f}")




        





