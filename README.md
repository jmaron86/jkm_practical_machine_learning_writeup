# jkm practical machine learning writeup

## installing packages and libraries
The following packages and libraries were used to divide, train, visualize and explain the data.
```
install.packages("caret")
install.packages("lattice")
install.packages("ggplot2")
library(ggplot2)
library(lattice)
library(caret)
```
## Loading the training set
```
d1<-read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
```
## dividing the training data into two dataframes
One dataframe is used for training the model and the other for testing the predictive nature of the model.
60% of the data is allocated to training.

```
inTrain<-createDataPartition(y=d1$classe,p=0.6,list=FALSE)
train1<-d1[inTrain,];test1<-d1[-inTrain,]
```
## Scrubbing and refining the training data
A visual inspection of the CSV training data revealed a significant portion of the 160 columns to be have predominantly blank values or have a preponderance of "NA".  This was my approach to cleaning the data and preparing it for a training model.
#####Accounting for columns with mostly empty values
I converted all empty cells to "NA".
```
train1[train1==""]<-NA
```
#####Removing columns with high incidence of "NA"
Then I removed any column from the training portion of the training set that had "NA" for over 80% of its total values.
```
train1<-train1[,colSums(is.na(train1))<nrow(train1)*0.8]
```
#####Miscellaneous adjustments to data
The index column (i.e. "X") was removed, as was the non-numeric timestamp.  This timestamp was superfluous because it was comprised of numeric data in two other columns.  
```
train1<-train1[c(-1)]
train1<-train1[c(-4)]
```
Columns identifying the user name and window were not in numeric format.  I adjusted these to allow for correlation analysis.
```
train1$user_name<-as.numeric(train1$user_name)
train1$new_window<-as.numeric(train1$new_window)
```
#####Identifying highly correlating variables and pruning dataset
Highly correlated variables can often be removed.  I determined which columns exhibited a correlation of at least 0.9.
```
tr2Corr<-cor(train2)
findCorrelation(tr2Corr, cutoff = .90, verbose = FALSE)
names(train2)[c(6,13,14,15,24,41)] 
```
I removed all but one of these correlated variables.
```
train1<-train1[c(-6)]
train1<-train1[c(-14)]
train1<-train1[c(-15)]
train1<-train1[c(-24)]
train1<-train1[c(-41)]
```
#####Removing the outcome variable
The model should not be trained on its own outcome variable and therefore I removed it prior to using any machine learning algorithms.
```
train2<-train1[c(-58)]
```
## Setting up parallel processing before training the model
I installed and loaded packages for parallel processing to expedite the computation time.
```
library(parallel)
install.packages("doParallel")
library(doParallel)
clus<-makeCluster(detectCores()-1)
registerDoParallel(clus)
```
### Designing a cross-validation training method 
Because I anticipated that a random forest model would yield the best results I decided [for ease of use] to simplify the baseline training method for quicker results. I chose a cross-validation approach that would create 5 folds of data.  Parallel processing is explicitly permitted here. 
```
fC<-trainControl(method="cv",number=5,allowParallel=TRUE)
```
## Applying a random forest training model 
Due to the complex, high-dimensional data I selected a random forest model to train the model using the training data and the training control method shown above.*
```
modFit<-train(classe~.,data=train1,method="rf",trControl=fC,prox=TRUE)  
```
##Inspecting the most accurate iteration
A random forest model is a variant of bootstrapping.  The 27th iteration was the most accurate and therefore serves as the basis for our model.
```
Resampling results across tuning parameters:

 | mtry | Accuracy | Kappa | Accuracy SD |    Kappa SD   | 
 | ---- | --------- | ----- | -------- | ------- |
 |  2   |   0.994   | 0.992 | 0.001511 | 0.00191 | 
 |  27  |   0.999   | 0.998 | 0.00076  | 0.000961|      
 |  52  |   0.995   | 0.994 | 0.00202  | 0.00256 |

Accuracy was used to select the optimal model using  the largest value.
The final value used for the model was mtry = 27. 
```
##Importance of variables in training model
Interpretability can be challenging when using random forest models, so I checked to see how the variables ranked in terms of their importance.  Surprisingly, the user was not among the 10 most sigificant predictors.
```
varImp(modFit)
rf variable importance

  only 10 most important variables shown (out of 52)

                     Overall
raw_timestamp_part_1 100.000
num_window            62.278
pitch_forearm         28.983
yaw_belt              25.623
magnet_dumbbell_z     23.247
pitch_belt            21.294
magnet_dumbbell_y     15.679
total_accel_belt      15.018
magnet_belt_z         10.935
roll_dumbbell          9.945
```
##In-sample error 
Because the model was fit to to this data, we would expect the accuracy to be higher than when we use the model to predict on a separate dataset.  Here we see 100 % accuracy, or no in-sample error.
```
pred0    A    B    C    D    E
    A 3348    0    0    0    0
    B    0 2279    0    0    0
    C    0    0 2054    0    0
    D    0    0    0 1930    0
    E    0    0    0    0 2165
```
## Modifying the test set accordingly
I made the same adjustments to the test dataset as with the training dataset.
```
test1[test1==""]<-NA
test1<-test1[,colSums(is.na(test1))<nrow(test1)*0.8]
test1<-test1[c(-1)]
test1<-test1[c(-4)]
test1$user_name<-as.numeric(test1$user_name)
test1$new_window<-as.numeric(test1$new_window)
test1<-test1[c(-6)]
test1<-test1[c(-14)]
test1<-test1[c(-15)]
test1<-test1[c(-24)]
test1<-test1[c(-41)]
```
##Making predictions on the test data
```
pred<-predict(modFit,test1)
test1$predRight<-pred==test1$classe
```
##Out-of-sample error
The table below illustrates the overall accuracy of our predictions for the classe variable based on our training model and test data set.  As expected, the out-of-sample error rate is higher than the in-sample rate, but is still acceptably low. 
```
table(pred,test1$classe)
pred    A    B    C    D    E
   A 2232    3    0    0    0
   B    0 1514    3    0    0
   C    0    0 1364    6    0
   D    0    1    1 1280    1
   E    0    0    0    0 1441
```
##Checking predictive accuracy on test dataset
The confusion matrix below illustrates the effectiveness of the random forest approach for this analysis.
```
confusionMatrix(pred,test1$classe)
```
```
Overall Statistics
                                          
               Accuracy : 0.9981          
                 95% CI : (0.9968, 0.9989)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                            
                  Kappa : 0.9976          
 Mcnemar's Test P-Value : NA              

Statistics by Class:
                            
                     Class: A  Class: B Class: C Class: D Class: E
Sensitivity            1.0000   0.9974   0.9971   0.9953   0.9993
Specificity            0.9995   0.9995   0.9991   0.9995   1.0000
Pos Pred Value         0.9987   0.9980   0.9956   0.9977   1.0000
Neg Pred Value         1.0000   0.9994   0.9994   0.9991   0.9998
Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
Detection Rate         0.2845   0.1930   0.1738   0.1631   0.1837
Detection Prevalence   0.2849   0.1933   0.1746   0.1635   0.1837
Balanced Accuracy      0.9997   0.9984   0.9981   0.9974   0.9997
```

## Evaluating model on independent test set
My code did not compile due to an issue with the CRAN mirrors.  The model had 100 % accuracy however.  The HTML and my results are included as an attachment.
