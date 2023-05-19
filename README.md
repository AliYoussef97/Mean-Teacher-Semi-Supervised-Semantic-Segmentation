
# Mean Teacher - Semi-Supervised Semantic Segmentation

This is an implementation of the Semi-Supervised Mean Teacher algorithm developed by Curious AI ([Paper](https://arxiv.org/abs/1703.01780),[Github Repository](https://github.com/CuriousAI/mean-teacher)).

A DeepLabv3 model was used, with a pre-trained Resnet50 backbone in this implementation. The model was trained on the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), with 20% labelled data (10% and 5% were also used, and found identical results to 20% labelled data).

It is advisable not to decrease the image size, as DeepLabv3 recommended minimum image size is [(224,224)](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/).
## Tunable Hyper-Parameters:

There are a few Hyper-Parameters that if further tuned, could potentially lead to better results:

- Learning-rate.
- Learning Rate Cosine Annealing Schedule Ramp-down.
- The optimizer used was Adam, however, RMSprop gave 
  very similar results, and could potentially be better than Adam if a detailed tuning is performed.
- EMA alpha
- Consistency, Mean-Teacher authors advise that if MSE is used for consistency loss, the consistency weight should be the number of classes or the number of classes squared.
- Consistency ramp-up (The number of epochs where consistency weight gradually increases)




## Dependencies

`torch`\
`torchvision`\
`torchmetrics`\
`segmentation_models_pytorch`\
`matplotlib`\
`PIL`\
`tqdm`

    
## Predicted Masks Examples


The following is an example of a predicted mask from the Oxford-IIIT Pet Dataset.

![alt text](https://github.com/AliYoussef97/Mean-Teacher-Semi-Supervised-Semantic-Segmentation/blob/main/Figures/Figure_2.png)

The model was tested on random animal images other than dogs and cats, and it performed fairly well. For example, the predicted mask for an Image containing Monkeys:

![alt text](https://github.com/AliYoussef97/Mean-Teacher-Semi-Supervised-Semantic-Segmentation/blob/main/Figures/Figure_3.png)

## Run

`...\Mean Teacher> python main.py`

Or simply open and run the main.py script in your favourite IDE.
## Authors

- [@AliYoussef](https://github.com/AliYoussef97)

