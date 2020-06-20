# CIFAR - ResNet18

## Objective

1. To draw cyclic triangule plot
2. To use custom resnet architecture as mentioned in instructions
3. To use one cycle policy with -
  i. Epoch - 24
  ii. Max at Epoch - 5
  iii. Find LR Min and LR Max
4. Target accuracy - 90%

## Result
1. Accuracy Achieved - Train : 98.51%, Test : 92.10%
2. LR Max - 0.046, LR Min - 0.00575

## Other Experiments
Max LR : 0.046

*   Min LR : 0.046/5 ==> Acc : 91.57
*   Min LR : 0.046/6 ==> Acc : 91.43
*   Min LR : 0.046/7 ==> Acc : 91.59
*   Min LR : 0.046/8 ==> Acc : 92.10
*   Min LR : 0.046/10 ==> Acc : 91.7

## Cyclic Triangle
![Cyclic triangle](https://github.com/santhiya-v/EVA/blob/master/S11/curve.png)

## Accuracy vs Learning Rate
![Acc LR](https://github.com/santhiya-v/EVA/blob/master/S11/acc_lr.png)

## Train and Test Accuracy
![train test accuracy](https://github.com/santhiya-v/EVA/blob/master/S11/train_test_acc.png)

## Misclassified Images
![Misclassified](https://github.com/santhiya-v/EVA/blob/master/S11/misclassified.png)

## GradCam Output
![Grad Cam](https://github.com/santhiya-v/EVA/blob/master/S11/gradCam.png)
