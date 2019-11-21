## Import the data and load libraries
library(caret)
train_data <- read.csv("train.csv", header = TRUE)

# Add a feature for tracking missing ages.
summary(train_data$Age)
train_data$MissingAge <- ifelse(is.na(train_data$Age),
                                "Y", "N")
#relabel Survived column as Yes or No instead of 1 and 0
train_data$Survived <- ifelse(train_data$Survived == "1",
                                "Y", "N")

# Set up factors.
train_data$Survived <- as.factor(train_data$Survived)
train_data$Pclass <- as.factor(train_data$Pclass)
train_data$Sex <- as.factor(train_data$Sex)
train_data$Embarked <- as.factor(train_data$Embarked)
train_data$MissingAge <- as.factor(train_data$MissingAge)


# Subset data to features we wish to keep/use.
# We remove name, ticket number, and fare since they do not help

features <- c("Survived", "Pclass", "Sex", "Age", "Embarked", "MissingAge")
train_data <- train_data[, features]
str(train_data)

## Split Data
# Create a 70%-30% split of the training data,
# keeping the proportions of the Survived class label the same across splits.

set.seed(12345)
indexes <- createDataPartition(train_data$Survived,
                               times = 1,
                               p = 0.7,
                               list = FALSE)
titanic.train_data <- train_data[indexes,]
titanic.test_data <- train_data[-indexes,]


## Impute Missing Ages

# Caret supports a number of mechanism for imputing (i.e., predicting) missing values. 
# Leverage bagged decision trees to impute missing values for the Age feature.

# First, transform all features of the training data to dummy variables.
dummy.vars <- dummyVars(~ ., data = titanic.train_data[, -1])
train.dummy <- predict(dummy.vars, titanic.train_data[, -1])
View(train.dummy)

# Imputation 
pre.process <- preProcess(train.dummy, method = "bagImpute")
imputed.data <- predict(pre.process, train.dummy)
View(imputed.data)

titanic.train_data$Age <- imputed.data[, 6]
View(titanic.train_data)

## Now we impute for our testing data
# First, transform all features of the testing data to dummy variables.
dummy.vars2 <- dummyVars(~ ., data = titanic.test_data[, -1])
test.dummy <- predict(dummy.vars2, titanic.test_data[, -1])
View(test.dummy)

# Imputation for testing data
pre.process2 <- preProcess(test.dummy, method = "bagImpute")
imputed.data2 <- predict(pre.process2, test.dummy)
View(imputed.data2)

titanic.test_data$Age <- imputed.data2[, 6]
View(titanic.test_data)









# Examine the proportions of the Survived class lable across the datasets.
prop.table(table(train_data$Survived))
prop.table(table(titanic.train_data$Survived))
prop.table(table(titanic.test_data$Survived))












##Tune parameters

fitControl <- trainControl(## 5-fold Cross Validation
  method = "repeatedcv",
  number = 5,
  ## repeated ten times
  repeats = 10,
  classProbs = TRUE)

##Apply different models to compare which performs the best

#Random Forests
set.seed(8947) # ensures paired resampling later
rffit <- train(Survived ~ ., data = titanic.train_data, 
               method = "ranger", 
               trControl = fitControl,
               verbose = FALSE)


#Support Vector Machine with Linear Kernel
set.seed(8947) # ensures paired resampling later
svmfit <- train(Survived ~ ., data = titanic.train_data, 
                method = "svmLinear", 
                trControl = fitControl,
                verbose = FALSE)


##Compare the results and visualize them
results <- resamples(list("Random.Forest"=rffit,"SVM"=svmfit))
summary(results) 
bwplot(results,scales=list(relation="free"))








## Now test on testing partition of our data

rf.test_data <- predict(rffit,titanic.test_data)
svm.test_data <- predict(svmfit,titanic.test_data)
train_data.results <- rbind(
    "Random Forest"=postResample(pred=rf.test_data,obs=titanic.test_data$Survived),
    "SVM"=postResample(pred=svm.test_data,obs=titanic.test_data$Survived)
  )
print(train_data.results)

## mess around with metric in train() - kappa or accuracy
## ROC requirement in fitControl - Try out more feature engineering
## play with tuning grid
## model more with logical regression & naivebayes
## fit the model with titanic.test_data & titanic.train_data
## then choose top 2-3 models and use real test data to see if they hold up

