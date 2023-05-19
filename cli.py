import argparse

def parser():
    parser = argparse.ArgumentParser(description='Mean teacher implementation parameters')
    parser.add_argument('--lr', default=0.0001, type=int,
                        help='Learning Rate')
    parser.add_argument('--lr_ramp_down', default=125, type=int,
                        help='Learning Rate ramp down')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of epochs')
    parser.add_argument('--img_size', default=256, type=int,
                        help='Image_size')
    parser.add_argument('--crop_size', default=224, type=int,
                        help='Crop_size')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch_size')
    parser.add_argument('--alpha', default=0.999, type=float,
                        help='Mean Teacher alpha parameter')
    parser.add_argument('--consistency_rampup', default=15, type=int,
                        help='Mean Teacher consistency_rampup parameter')
    parser.add_argument('--consistency', default=4, type=int,
                        help='Mean Teacher consistency parameter')
    parser.add_argument('--global_step', default=0, type=int,
                        help='Mean Teacher global_step parameter')
    parser.add_argument('--ratio_data_training', default=0.70, type=float,
                        help='Ratio of training data from full data')
    parser.add_argument('--ratio_data_validation', default=0.20, type=float,
                        help='Ratio of Validation data from full data')
    parser.add_argument('--ratio_training_laballed', default=0.20, type=float,
                        help='Ratio of laballed data from training data')
    parser.add_argument('--full_data_path', default="./data/oxford-iiit-pet/annotations/list.txt", type=str,
                        help='Path of full dataset names and annotations')
    parser.add_argument('--img_path', default="./data/oxford-iiit-pet/images", type=str,
                        help='Path to images.')
    parser.add_argument('--seg_path', default="./data/oxford-iiit-pet/annotations/trimaps/", type=str,
                        help='Path to trimaps')
   
    # Change to parser.parse_args("") if you are using a cloud notebook environment like Google Colab or Kaggle.
    return parser.parse_args()
