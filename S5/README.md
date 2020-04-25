# S5 Assignment

## Goal 
To train MNIST model and achieve test accuracy of 99.4%.

### Constraints
1. In 15 epochs
2. Parameters should be less than 10K

Following were the steps followed to achieve the goal

## Step 1

### Target	
Getting basic code setup - up and running	
Setting the basic skeleton right	
### Result	
Parameters: 194k	
Best Train Accuracy: 99.36	
Best Test Accuracy: 99.23	
### Analysis	
Heavy model	
Long way to reach the goal	

## Step 2

### Target	
Reducing the number of parameters so we are nearing our goal of less than 10K parameters	
### Result	
Parameters: 8K	
Best Train Accuracy: 98.71	
Best Test Accuracy: 98.57	
### Analysis	
Model is not over-fitting	
Reducing the number of parameters, have reduced accuracy little	

## Step 3

### Target
Adding Batch-Normalization
Adding regularization Drop out to each layer (of 10%)
### Result
Parameters: 8K
Best Train Accuracy: 99.15
Best Test Accuracy: 99.13
### Analysis
Adding Batch-Norm and Drop out has significantly increased both train and test accuracy
Model is not overfitting which is a good thing

## Step 4

### Target	
Adding GAP layer	
Increasing the capacity towards end of the prediction especially by adding layer after GAP.	
### Result	
Parameters: 9,752	
Best Train Accuracy: 99.11	
Best Test Accuracy: 99.20	
### Analysis	
Adding GAP has increased model accuracy.	
Since Test accuracy is greater than train accuracy, this could be under-fitting model	

## Step 5

### Target
Adding random rotation of 7 degrees to the images
Added LR scheduler
### Result
Parameters: 9,752
Best Train Accuracy: 98.94
Best Test Accuracy: 99.20
### Analysis
Accuracy is switching between 99.30-99.20
Model seems to be slightly under- fitting
Image rotation, has reduced the train accuracy however increased test accuracy

