# Prediction Model for Weight Lifting Exercises Quality

This work is done as a part of course project for "Practical Machine Learning" course at Coursera. The goal of the assignment is to build prediction model for exercise from "Weight Lifting Exercises" dataset (available at http://groupware.les.inf.puc-rio.br/har). 

## Loading Data
Load libraries that will be used for prediction models:


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
# Ensure reproducibility
set.seed(201502)
```

Load data and make sure missing values are marked properly as N/A:


```r
# List of missing values 
naStrings = c("", "NA", "#DIV/0!")
dataset = read.csv("pml-training.csv", header = TRUE, na.strings = naStrings)
validation = read.csv("pml-testing.csv", header = TRUE, na.strings = naStrings)
```

Checking summmary of loaded training data we see that there are many N/A values there. In order to clean them, we will remove column that contain more than a half of N/A values. In addition first 7 columns are removed, since they represent personal data and can be inadvertently picked by prediction model which could lead to overfitting: 


```r
threshold = dim(dataset)[1] / 2
cleanDataset = dataset[, apply(dataset, 2, function(x) sum(is.na(x)) < threshold)]
removeColumns = 1:7
cleanDataset = cleanDataset[, -removeColumns]

cat('Number of columns in raw training data: ', dim(dataset)[2])
```

```
## Number of columns in raw training data:  160
```

```r
cat('Number of columns in cleaned training data: ', dim(cleanDataset)[2])
```

```
## Number of columns in cleaned training data:  53
```

```r
rm(dataset)
```

## Create Training and Testing Datasets

To train the prediction model we will split the training data into training and testing subsets:


```r
inTraining = createDataPartition(y = cleanDataset$classe, p = 0.7, list = FALSE)
training = cleanDataset[inTraining, ]
testing = cleanDataset[-inTraining, ]
```

## Building the Prediction Model

We will try to build prediction model using Random Forests (method "rf" in train() function, but via randomForest() function for performance reasons):


```r
modFit = randomForest(classe ~ ., data = training)
modFit
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.5%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3902    4    0    0    0 0.001024066
## B   14 2639    5    0    0 0.007148232
## C    0   13 2380    3    0 0.006677796
## D    0    0   21 2231    0 0.009325044
## E    0    0    2    6 2517 0.003168317
```

Model shows very low error rate estiamate. We will check the model by comparing prediction results in testing dataset:


```r
pred = predict(modFit, testing)
confusionMatrix(pred, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    4    0    0    0
##          B    0 1135    8    0    0
##          C    0    0 1018    7    0
##          D    0    0    0  953    2
##          E    1    0    0    4 1080
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9956          
##                  95% CI : (0.9935, 0.9971)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9944          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9965   0.9922   0.9886   0.9982
## Specificity            0.9991   0.9983   0.9986   0.9996   0.9990
## Pos Pred Value         0.9976   0.9930   0.9932   0.9979   0.9954
## Neg Pred Value         0.9998   0.9992   0.9984   0.9978   0.9996
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1929   0.1730   0.1619   0.1835
## Detection Prevalence   0.2850   0.1942   0.1742   0.1623   0.1844
## Balanced Accuracy      0.9992   0.9974   0.9954   0.9941   0.9986
```

The above test might not be necessary, as in "Random Forests" by Leo Breiman and Adele Cutler (availabe at http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#ooberr) authors state that test error is estimated internally, and therefore there is no need for cross-validation.

## Submission Results

For completing submission part of the project we will create solution files, one per each answer:


```r
submission = predict(modFit, validation)

write_files = function(x) {
    n = length(x)
    for(i in 1:n) {
        filename = sprintf("problem_%02i.txt", i)
        write.table(x[i], file = filename, 
                    quote = FALSE, row.names = FALSE, col.names = FALSE)
    }
}
write_files(submission)
```
