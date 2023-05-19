import torch
from data import all_class_dictionary,label_val_split, download_data, Data
from torch.utils.data import DataLoader
from mt_functions import student_teacher_models
from train_val import train_val
from cli import parser
from test import test

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    download_data()

    args = parser()
    
    all_classes = all_class_dictionary(args.full_data_path)

    train_names, validation_names, test_names, laballed_train_names = label_val_split(all_classes, args.ratio_data_training, args.ratio_data_validation, args.ratio_training_laballed)

    train_data = Data(train_names, args.img_path, args.seg_path, args.img_size, args.crop_size, unlaballed_flag = True, laballed_names = laballed_train_names)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True)

    validation_data = Data(validation_names, args.img_path, args.seg_path, args.img_size, args.crop_size)
    validation_loader = DataLoader(validation_data, args.batch_size, shuffle=False)

    student, teacher = student_teacher_models(device)

    train_val(student, teacher, train_loader, validation_loader, device, args)

    test(test_names, student, device, args)