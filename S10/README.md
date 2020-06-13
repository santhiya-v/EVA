# CIFAR - ResNet18

## Objective
1. To use CutOut as Image augmentation technique
2. To use LR finder to find best LR range suitable for the model
3. To achieve 88% accuracy
4. To use SDG with Momentum
5. GradCam on 25 misclassified images

## Result
1. Accuarcy Achieved - 87.37%
2. Best LR range for the model - 0.5 to 0.005
3. Data transformation used - ToTensor, Normalize, HorizontalFlip, CutOut
4. No. of Epochs - 50
5. Loss/Regularization - L2


## Train and Test Accuracy
![train test accuracy](https://github.com/santhiya-v/EVA/blob/master/S10/train_test_accuracy.png)

## GradCam Output on 25 Misclassified Images
![gradcam images](https://github.com/santhiya-v/EVA/blob/master/S10/gradcam.png)
