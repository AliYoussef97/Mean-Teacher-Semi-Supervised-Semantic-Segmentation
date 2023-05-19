import random
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import datasets
import torchvision.transforms.functional as F
import random
from torchvision import transforms

class Data(Dataset):
    """
    Data class:
        
        - Data class inherits from torch Dataset.
         
        - It takes class_list, the images and trimaps path, image size and crop size, unlaballed flag and
          names of laballed data in training set.

        - In the constrtuor, the labelled_class_names list is created with only the names without annotations,
          and the data is loaded to memory.
        
        - augmentation function applies augmentaions on images and blend_mask function sets 
          background and unknown to 0, forground to 1.
        
        - In  __getitem__, the images and tripamps are fetched, the augmentations are applied on images and trimaps.
        
        - If the unlaballed flag is true, a second image of the same image is created with different noise from
          the augmentation function.

        - If then check if the cucrent image is in the labaled_class_names, if it is not, the labels are set to 
          -1 indicating it is unlaballed, if it is, the trimaps are returnd.

        - If the unlaballed flag is false, then the image and trimap are returnd, this is for validation dataset.

    Inputs:
        - class_list: List containing a pet names, type: list

        - images_path: The path to images folder.

        - trimap_path: The path to trimap folder.

        - image_size: The desired image size.

        - crop_size: The desired crop size.

        - ratio_labelled: Ratio of training data that is laballed.

        - unlaballed_flag: True for laballed/unlaballed data (for training), False if not (validation).

    Outputs:
        - images/trimaps: 2 images with different noise and mask for training data, and 1 image without noise and mask for validation data.
    
    """
    def __init__(self, class_list, images_path, trimap_path, img_size, crop_size, unlaballed_flag = False, laballed_names = None):
    

        self.class_list = class_list
        self.images_path = images_path
        self.trimap_path = trimap_path
        self.unlaballed_flag = unlaballed_flag
        self.img_size = img_size
        self.crop_size = crop_size
        self.laballed_names = laballed_names
        
        if unlaballed_flag == True:

            self.labelled_class_names = [c[0] for c in self.laballed_names]


        self.data = []

        for c in class_list:

            image = os.path.join(self.images_path, c[0] + ".jpg")
            trimap_image = os.path.join(self.trimap_path, c[0] + ".png")

            RGB_image = Image.open(image).convert("RGB")
            trimap_image = Image.open(trimap_image)

            self.data.append([RGB_image,trimap_image])

    
    @staticmethod
    def augmentation(image, trimap, img_size, crop_size, flag = "train"):

        
        resize = transforms.Resize((img_size,img_size))
        center_crop = transforms.CenterCrop(crop_size)
        to_tensor = transforms.ToTensor()
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        noise = transforms.Compose([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                    transforms.GaussianBlur(3,sigma=(0.1,2.0))])
      

        if flag == "train":
            
            i, j, h, w = transforms.RandomResizedCrop.get_params(image,scale=(0.08,1.0),ratio=(3/4,4/3))
            image = F.resized_crop(image,i,j,h,w,size=(crop_size,crop_size))
            trimap = F.resized_crop(trimap,i,j,h,w,size=(crop_size,crop_size))

            if random.random() < 0.75:
                angle = random.randint(-10,10)
                image = F.rotate(image,angle)
                trimap = F.rotate(trimap,angle)
    
            if random.random() < 0.75:
                Hflip = transforms.RandomHorizontalFlip(p=1)
                image = Hflip(image)
                trimap = Hflip(trimap)
            
            train_image1 = noise(image)
            train_image2 = noise(image)
                
            train_image1 = norm(to_tensor(train_image1))
            train_image2 = norm(to_tensor(train_image2))
            train_trimap = to_tensor(trimap)

            return train_image1, train_image2, train_trimap
        
        elif flag == "val":
            
            val_image = resize(image)
            val_trimap = resize(trimap)

            val_image = center_crop(val_image)
            val_trimap = center_crop(val_trimap)

            val_image = norm(to_tensor(val_image))
            val_trimap = to_tensor(val_trimap)
            
            return val_image, val_trimap
        
        
    @staticmethod
    def mask_blend(trimap_image):
        
        trimap_image[trimap_image == (2.0 / 255)] = 0.0
        trimap_image[trimap_image == (3.0 / 255)] = 0.0     
        trimap_image[trimap_image == (1.0 / 255)] = 1.0
        
        return trimap_image

    def __len__(self):

        return len(self.class_list)

    def __getitem__(self, idx):

        current_image, current_trimap =  self.data[idx]

        if self.unlaballed_flag:
    
            dataset_image, dataset_image2, trimap_image = self.augmentation(current_image, current_trimap, self.img_size, self.crop_size, "train")

            if self.class_list[idx][0] not in self.labelled_class_names:
                trimap_image = torch.full(trimap_image.size(), -1)
                
            else:
                trimap_image = self.mask_blend(trimap_image)

            return dataset_image, dataset_image2, trimap_image
  
        else:

            dataset_image, trimap_image = self.augmentation(current_image, current_trimap, self.img_size, self.crop_size, "val")
            trimap_image = self.mask_blend(trimap_image)

            return dataset_image, trimap_image


def all_class_dictionary(full_data_path):

    """ 
    all_class_dictionary:

        - The function creates a key from a pet name if not present, the values of the keys are then the full name of the pet
          and the pixel annotations. ex.{"american_bulldog": ('american_bulldog_100', 2, 2, 1), ('american_bulldog_101', 2, 2, 1)}

    Inputs:

        -None

    Outputs:
        
        - data_dict: A dictionary containing class names, type: dictionary

    
    """
    
    data_dict = {}

    with open(full_data_path, 'r') as f:
        for line in f: 
            if line.startswith("#"):
                    continue
            
            name, num1, num2, num3 = line.split()

            class_name = "_".join(name.split("_")[:-1])

            if class_name not in data_dict:
                data_dict[class_name] = []
            
            data_dict[class_name].append((name, int(num1), int(num2), int(num3)))
    

    return data_dict


def label_val_split(data_dict,train_ratio,validation_ratio,ratio_labelled):

    """
    label_val_split:

        - This function takes a class containing all pet names where the pet name is the class key,
          and all names related to that class are the values.

        - It loops over the values of each class, shuffles and choose and splits the names of the class
          into training and validation.
    
    - Inputs:

        - data_dict: dictionary containing class names, type: dictionary

        - train_ratio: training split, type: float

        - validation_ratio: validation split, type: float

    - Outputs:

        - training_data: A list containing names that will be extracted for the training dataset, type: list

        - validation: A list containing names that will be extracted for the validation dataset, type: list

        - test_data: A list containing names that will be extracted for the test dataset, type: list
    """
    
    training_data = []
    laballed_train_data =[]
    validation = []
    test_data = []

    for _, class_data in data_dict.items():

        num_data = len(class_data) 

        num_training = int(train_ratio * num_data)
        num_validation = int(validation_ratio * num_data) 
        num_test = num_data - num_training - num_validation

        random.shuffle(class_data)  

        training_data.extend(class_data[:num_training])  
        validation.extend(class_data[num_training:num_training+num_validation])
        test_data.extend(class_data[-num_test:])

        num_laballed_per_class = int(ratio_labelled * len(class_data[:num_training]))
        labelled_names = random.sample(class_data[:num_training], num_laballed_per_class)
        laballed_train_data.extend(labelled_names)

    return training_data, validation, test_data, laballed_train_data


def download_data():
    datasets.OxfordIIITPet(root='./data',download=True)
    return