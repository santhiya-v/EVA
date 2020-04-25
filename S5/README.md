# S5 Assignment

## Goal 
To train MNIST model and achieve test accuracy of 99.4%.

### Constraints
1. In 15 epochs
2. Parameters should be less than 10K

Following were the steps followed to achieve the goal

## Step 1

### Target	
1. Getting basic code setup - up and running	
2. Setting the basic skeleton right	
### Result	
1. Parameters: 194k	
2. Best Train Accuracy: 99.36	
3. Best Test Accuracy: 99.23	
### Analysis	
1. Heavy model	
2. Long way to reach the goal	

## Step 2

### Target	
1. Reducing the number of parameters so we are nearing our goal of less than 10K parameters	
### Result	
1. Parameters: 8K	
2. Best Train Accuracy: 98.71	
3. Best Test Accuracy: 98.57	
### Analysis	
1. Model is not over-fitting	
2. Reducing the number of parameters, have reduced accuracy little	

## Step 3

### Target
1. Adding Batch-Normalization
2. Adding regularization Drop out to each layer (of 10%)
### Result
1. Parameters: 8K
2. Best Train Accuracy: 99.15
3. Best Test Accuracy: 99.13
### Analysis
1. Adding Batch-Norm and Drop out has significantly increased both train and test accuracy
2. Model is not overfitting which is a good thing

## Step 4

### Target	
1. Adding GAP layer	
2. Increasing the capacity towards end of the prediction especially by adding layer after GAP.	
### Result	
1. Parameters: 9,752	
2. Best Train Accuracy: 99.11	
3. Best Test Accuracy: 99.20	
### Analysis	
1. Adding GAP has increased model accuracy.	
2. Since Test accuracy is greater than train accuracy, this could be under-fitting model	

## Step 5

### Target
1. Adding random rotation of 7 degrees to the images
2. Added LR scheduler
### Result
1. Parameters: 9,752
2. Best Train Accuracy: 98.94
3. Best Test Accuracy: 99.20
### Analysis
1. Accuracy is switching between 99.30-99.20
2. Model seems to be slightly under- fitting
3. Image rotation, has reduced the train accuracy however increased test accuracy

