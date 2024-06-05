library(caret)
library(klaR)
library(ipred)
library(rpart)
library(randomForest)

setwd("D:/STUDIA/Sem_9/Data_mining/Project")

my_labels <- c("ID", "Diagnosis", "Radius1", "Texture1", "Perimeter1", "Area1", "Smoothness1", "Compactness1", "Concavity1", "Concave_points1", "Symmetry1", "Fractal_dimension1",
                                  "Radius2", "Texture2", "Perimeter2", "Area2", "Smoothness2", "Compactness2", "Concavity2", "Concave_points2", "Symmetry2", "Fractal_dimension2",
                                  "Radius3", "Texture3", "Perimeter3", "Area3", "Smoothness3", "Compactness3", "Concavity3", "Concave_points3", "Symmetry3", "Fractal_dimension3")

data <- read.csv(file="wdbc.data", header = FALSE, col.names=my_labels, stringsAsFactors = TRUE)
data <- data[,-1]

#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
# Classification
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------

# Check correlation
corrFeat.id <- findCorrelation(cor(data[,-1]))
corrFeat.names <- findCorrelation(cor(data[,-1]), names=T)

# delete correlation
data2 <- data[, -c(corrFeat.id+1)]

#------------------------------------------------------------------------------------
# Learning and test set split
#------------------------------------------------------------------------------------

# We randomly select portion of data (2/3 of all observations) for the learning set,
# and assign the remaining observations to the test set
n <- 569
prop <- 2/3

set.seed(250) #set the generator seed in order to obtain reproducible results

learning.indx <- sample(1:n, prop*n)
learning.set <- data2[learning.indx,]
test.set <- data2[-learning.indx,]

# check class proportion
prop.table(table(learning.set[,1]))
prop.table(table(test.set[,1]))

prop.table(table(data2[,1]))

# Feature selection - stepwise method
# lda.forward.selection <- stepclass(Diagnosis~., data=learning.set, method="lda", direction="forward", improvement=0.0001)
# qda.forward.selection <- stepclass(Diagnosis~., data=learning.set, method="qda", direction="forward", improvement=0.0001)
# knn.forward.selection <- stepclass(Diagnosis~., data=learning.set, method="sknn", direction="forward", improvement=0.0001)

# 4 features sets
# 1. Concave_points3+Area1+Texture3+Radius2+Fractal_dimension2 
# 2. Concave_points3+Area1+Texture3+Concave_points2+Smoothness3 
# 3. Concave_points3+Radius2+Concavity3+Compactness1
# 4. all after deleting correlations

# Construction of the classification rule based on selected variables
data.lda1 <- lda(Diagnosis~Concave_points3+Area1+Texture3+Radius2+Fractal_dimension2, data=learning.set)
data.qda1 <- qda(Diagnosis~Concave_points3+Area1+Texture3+Radius2+Fractal_dimension2, data=learning.set)
data.knn1 <- ipredknn(Diagnosis~Concave_points3+Area1+Texture3+Radius2+Fractal_dimension2, data=learning.set, k=5)
data.logit1 <- glm(Diagnosis~Concave_points3+Area1+Texture3+Radius2+Fractal_dimension2, data=learning.set, family=binomial(link="logit"))
data.tree1 <- rpart(Diagnosis~Concave_points3+Area1+Texture3+Radius2+Fractal_dimension2, data=learning.set)
data.bagging1 <- bagging(Diagnosis~Concave_points3+Area1+Texture3+Radius2+Fractal_dimension2, data=learning.set)
data.randomForest1 <- randomForest(Diagnosis~Concave_points3+Area1+Texture3+Radius2+Fractal_dimension2, data=learning.set)

data.lda2 <- lda(Diagnosis~Concave_points3+Area1+Texture3+Concave_points2+Smoothness3, data=learning.set)
data.qda2 <- qda(Diagnosis~Concave_points3+Area1+Texture3+Concave_points2+Smoothness3, data=learning.set)
data.knn2 <- ipredknn(Diagnosis~Concave_points3+Area1+Texture3+Concave_points2+Smoothness3, data=learning.set, k=5)
data.logit2 <- glm(Diagnosis~Concave_points3+Area1+Texture3+Concave_points2+Smoothness3, data=learning.set, family=binomial(link="logit"))
data.tree2 <- rpart(Diagnosis~Concave_points3+Area1+Texture3+Concave_points2+Smoothness3, data=learning.set)
data.bagging2 <- bagging(Diagnosis~Concave_points3+Area1+Texture3+Concave_points2+Smoothness3, data=learning.set)
data.randomForest2 <- randomForest(Diagnosis~Concave_points3+Area1+Texture3+Concave_points2+Smoothness3, data=learning.set)

data.lda3 <- lda(Diagnosis~Concave_points3+Radius2+Concavity3+Compactness1, data=learning.set)
data.qda3 <- qda(Diagnosis~Concave_points3+Radius2+Concavity3+Compactness1, data=learning.set)
data.knn3 <- ipredknn(Diagnosis~Concave_points3+Radius2+Concavity3+Compactness1, data=learning.set, k=5)
data.logit3 <- glm(Diagnosis~Concave_points3+Radius2+Concavity3+Compactness1, data=learning.set, family=binomial(link="logit"))
data.tree3 <- rpart(Diagnosis~Concave_points3+Radius2+Concavity3+Compactness1, data=learning.set)
data.bagging3 <- bagging(Diagnosis~Concave_points3+Radius2+Concavity3+Compactness1, data=learning.set)
data.randomForest3 <- randomForest(Diagnosis~Concave_points3+Radius2+Concavity3+Compactness1, data=learning.set)

data.lda4 <- lda(Diagnosis~., data=learning.set)
data.qda4 <- qda(Diagnosis~., data=learning.set)
data.knn4 <- ipredknn(Diagnosis~., data=learning.set, k=5)
data.logit4 <- glm(Diagnosis~., data=learning.set, family=binomial(link="logit"))
data.tree4 <- rpart(Diagnosis~., data=learning.set)
data.bagging4 <- bagging(Diagnosis~., data=learning.set)
data.randomForest4 <- randomForest(Diagnosis~., data=learning.set)

# prediction for the test set
prediction.logit1  <-  predict(data.logit1, test.set[,-1])
prediction.logit2  <-  predict(data.logit2, test.set[,-1])
prediction.logit3  <-  predict(data.logit3, test.set[,-1])
prediction.logit4  <-  predict(data.logit4, test.set[,-1])

# auxiliary function to convert probabilities to class labels for a given cutoff
prob2labels <- function(probs, cutoff)
{
  classes <- rep("B",length(probs))
  classes[probs>cutoff] <- "M"
  return(as.factor(classes))
}

# predicted class labels
pred.labels.lda1 <- predict(data.lda1, test.set[,-1])$class
pred.labels.qda1 <- predict(data.qda1, test.set[,-1])$class
pred.labels.knn1 <- predict(data.knn1, test.set[,-1], type="class")
pred.labels.logit1 <- prob2labels(probs=prediction.logit1, cutoff=0.5)
pred.labels.tree1 <- predict(data.tree1, test.set[,-1], type="class")
pred.labels.bagging1 <- predict(data.bagging1, test.set[,-1], type="class")
pred.labels.randomForest1 <- predict(data.randomForest1, test.set[,-1], type="class")

pred.labels.lda2 <- predict(data.lda2, test.set[,-1])$class
pred.labels.qda2 <- predict(data.qda2, test.set[,-1])$class
pred.labels.knn2 <- predict(data.knn2, test.set[,-1], type="class")
pred.labels.logit2 <- prob2labels(probs=prediction.logit2, cutoff=0.5)
pred.labels.tree2 <- predict(data.tree2, test.set[,-1], type="class")
pred.labels.bagging2 <- predict(data.bagging2, test.set[,-1], type="class")
pred.labels.randomForest2 <- predict(data.randomForest2, test.set[,-1], type="class")

pred.labels.lda3 <- predict(data.lda3, test.set[,-1])$class
pred.labels.qda3 <- predict(data.qda3, test.set[,-1])$class
pred.labels.knn3 <- predict(data.knn3, test.set[,-1], type="class")
pred.labels.logit3 <- prob2labels(probs=prediction.logit3, cutoff=0.5)
pred.labels.tree3 <- predict(data.tree3, test.set[,-1], type="class")
pred.labels.bagging3 <- predict(data.bagging3, test.set[,-1], type="class")
pred.labels.randomForest3 <- predict(data.randomForest3, test.set[,-1], type="class")

pred.labels.lda4 <- predict(data.lda4, test.set[,-1])$class
pred.labels.qda4 <- predict(data.qda4, test.set[,-1])$class
pred.labels.knn4 <- predict(data.knn4, test.set[,-1], type="class")
pred.labels.logit4 <- prob2labels(probs=prediction.logit4, cutoff=0.5)
pred.labels.tree4 <- predict(data.tree4, test.set[,-1], type="class")
pred.labels.bagging4 <- predict(data.bagging4, test.set[,-1], type="class")
pred.labels.randomForest4 <- predict(data.randomForest4, test.set[,-1], type="class")

# classification accuracy assessment: confusion matrix, misclassification error
real.labels <- test.set[, 1] # real labels for objects from the test set
conf.mat.lda1 <- table(pred.labels.lda1, real.labels) 
conf.mat.qda1 <- table(pred.labels.qda1, real.labels)
conf.mat.knn1 <- table(pred.labels.knn1, real.labels)
conf.mat.logit1 <- table(pred.labels.logit1, real.labels)
conf.mat.tree1 <- table(pred.labels.tree1, real.labels)
conf.mat.bagging1 <- table(pred.labels.bagging1, real.labels)
conf.mat.randomForest1 <- table(pred.labels.randomForest1, real.labels)

conf.mat.lda2 <- table(pred.labels.lda2, real.labels) 
conf.mat.qda2 <- table(pred.labels.qda2, real.labels)
conf.mat.knn2 <- table(pred.labels.knn2, real.labels)
conf.mat.logit2 <- table(pred.labels.logit2, real.labels)
conf.mat.tree2 <- table(pred.labels.tree2, real.labels)
conf.mat.bagging2 <- table(pred.labels.bagging2, real.labels)
conf.mat.randomForest2 <- table(pred.labels.randomForest2, real.labels)

conf.mat.lda3 <- table(pred.labels.lda3, real.labels) 
conf.mat.qda3 <- table(pred.labels.qda3, real.labels)
conf.mat.knn3 <- table(pred.labels.knn3, real.labels)
conf.mat.logit3 <- table(pred.labels.logit3, real.labels)
conf.mat.tree3 <- table(pred.labels.tree3, real.labels)
conf.mat.bagging3 <- table(pred.labels.bagging3, real.labels)
conf.mat.randomForest3 <- table(pred.labels.randomForest3, real.labels)

conf.mat.lda4 <- table(pred.labels.lda4, real.labels) 
conf.mat.qda4 <- table(pred.labels.qda4, real.labels)
conf.mat.knn4 <- table(pred.labels.knn4, real.labels)
conf.mat.logit4 <- table(pred.labels.logit4, real.labels)
conf.mat.tree4 <- table(pred.labels.tree4, real.labels)
conf.mat.bagging4 <- table(pred.labels.bagging4, real.labels)
conf.mat.randomForest4 <- table(pred.labels.randomForest4, real.labels)


# classification error on the test set
n.test <- dim(test.set)[1]   #number of objects in the test set
error.lda1 <- (n.test-sum(diag(conf.mat.lda1)))/n.test
error.qda1 <- (n.test-sum(diag(conf.mat.qda1)))/n.test
error.knn1 <- (n.test-sum(diag(conf.mat.knn1)))/n.test
error.logit1 <- (n.test-sum(diag(conf.mat.logit1)))/n.test
error.tree1 <- (n.test-sum(diag(conf.mat.tree1)))/n.test
error.bagging1 <- (n.test-sum(diag(conf.mat.bagging1)))/n.test
error.randomForest1 <- (n.test-sum(diag(conf.mat.randomForest1)))/n.test

error.lda2 <- (n.test-sum(diag(conf.mat.lda2)))/n.test
error.qda2 <- (n.test-sum(diag(conf.mat.qda2)))/n.test
error.knn2 <- (n.test-sum(diag(conf.mat.knn2)))/n.test
error.logit2 <- (n.test-sum(diag(conf.mat.logit2)))/n.test
error.tree2 <- (n.test-sum(diag(conf.mat.tree2)))/n.test
error.bagging2<- (n.test-sum(diag(conf.mat.bagging2)))/n.test
error.randomForest2 <- (n.test-sum(diag(conf.mat.randomForest2)))/n.test

error.lda3 <- (n.test-sum(diag(conf.mat.lda3)))/n.test
error.qda3 <- (n.test-sum(diag(conf.mat.qda3)))/n.test
error.knn3 <- (n.test-sum(diag(conf.mat.knn3)))/n.test
error.logit3 <- (n.test-sum(diag(conf.mat.logit3)))/n.test 
error.tree3 <- (n.test-sum(diag(conf.mat.tree3)))/n.test
error.bagging3 <- (n.test-sum(diag(conf.mat.bagging3)))/n.test
error.randomForest3 <- (n.test-sum(diag(conf.mat.randomForest3)))/n.test

error.lda4 <- (n.test-sum(diag(conf.mat.lda4)))/n.test
error.qda4 <- (n.test-sum(diag(conf.mat.qda4)))/n.test
error.knn4 <- (n.test-sum(diag(conf.mat.knn4)))/n.test
error.logit4 <- (n.test-sum(diag(conf.mat.logit4)))/n.test
error.tree4 <- (n.test-sum(diag(conf.mat.tree4)))/n.test
error.bagging4 <- (n.test-sum(diag(conf.mat.bagging4)))/n.test
error.randomForest4 <- (n.test-sum(diag(conf.mat.randomForest4)))/n.test

errors <- matrix(c(error.lda1, error.qda1, error.knn1, error.logit1, error.tree1, error.bagging1, error.randomForest1, error.lda2, error.qda2, error.knn2, error.logit2, error.tree2, error.bagging2, error.randomForest2, error.lda3, error.qda3, error.knn3, error.logit3, error.tree3, error.bagging3, error.randomForest3, error.lda4, error.qda4, error.knn4, error.logit4, error.tree4, error.bagging4, error.randomForest4), nrow=7, ncol=4, dimnames =list(c("error.lda", "error.qda", "error.knn",  "error.logit", "error.tree", "error.bagging", "error.randomForest"), c("1 features subset", "2 features subset","3 features subset", "4 features subset")))
errors <- round(errors,5)
accuracy <- (1-errors)*100
rownames(accuracy) <- c("accuracy.lda", "accuracy.qda", "accuracy.knn",  "accuracy.logit", "accuracy.tree", "accuracy.bagging", "accuracy.randomForest")

#------------------------------------------------------------------------------------
# Cross-validation
#------------------------------------------------------------------------------------

my.ipredknn <- function(formula1, data1, n.of.neighbors) ipredknn(formula=formula1, data=data1, k=n.of.neighbors)
my.glm <- function(formula1, data1, family1) glm(formula=formula1, data=data1, family=family1)
my.predict.lqda  <- function(model, newdata) predict(model, newdata=newdata)$class
my.predict.knn  <- function(model, newdata) predict(model, newdata=newdata, type="class")
my.predict.glm  <- function(model, newdata) prob2labels(probs=predict(model, newdata=newdata), cutoff=0.5)
mypredict.rpart <- function(object, newdata)  predict(object, newdata=newdata, type="class")

# comparison of classification errors: cv
error.lda1.cv <- errorest(Diagnosis~Concave_points3+Area1+Texture3+Radius2+Fractal_dimension2, data2, model=lda, predict=my.predict.lqda, estimator="cv", est.para=control.errorest(k = 10))$error
error.qda1.cv <- errorest(Diagnosis~Concave_points3+Area1+Texture3+Radius2+Fractal_dimension2, data2, model=qda, predict=my.predict.lqda, estimator="cv", est.para=control.errorest(k = 10))$error
error.knn1.cv <- errorest(Diagnosis~Concave_points3+Area1+Texture3+Radius2+Fractal_dimension2, data2, model=my.ipredknn, predict=my.predict.knn, estimator="cv", est.para=control.errorest(k = 10), n.of.neighbors=5)$error
error.logit1.cv <- errorest(Diagnosis~Concave_points3+Area1+Texture3+Radius2+Fractal_dimension2, data2, model=my.glm, predict=my.predict.glm, estimator="cv", est.para=control.errorest(k = 10), family1=binomial(link="logit"))$error
error.tree1.cv <- errorest(Diagnosis~Concave_points3+Area1+Texture3+Radius2+Fractal_dimension2, data2, model=rpart, predict=mypredict.rpart)$error
error.bagging1.cv <- errorest(Diagnosis~Concave_points3+Area1+Texture3+Radius2+Fractal_dimension2, data2, model=bagging)$error
error.randomForest1.cv <- errorest(Diagnosis~Concave_points3+Area1+Texture3+Radius2+Fractal_dimension2, data2, model=randomForest)$error

error.lda2.cv <- errorest(Diagnosis~Concave_points3+Area1+Texture3+Concave_points2+Smoothness3, data2, model=lda, predict=my.predict.lqda, estimator="cv", est.para=control.errorest(k = 10))$error
error.qda2.cv <- errorest(Diagnosis~Concave_points3+Area1+Texture3+Concave_points2+Smoothness3, data2, model=qda, predict=my.predict.lqda, estimator="cv", est.para=control.errorest(k = 10))$error
error.knn2.cv <- errorest(Diagnosis~Concave_points3+Area1+Texture3+Concave_points2+Smoothness3, data2, model=my.ipredknn, predict=my.predict.knn, estimator="cv", est.para=control.errorest(k = 10), n.of.neighbors=5)$error
error.logit2.cv <- errorest(Diagnosis~Concave_points3+Area1+Texture3+Concave_points2+Smoothness3, data2, model=my.glm, predict=my.predict.glm, estimator="cv", est.para=control.errorest(k = 10), family1=binomial(link="logit"))$error
error.tree2.cv <- errorest(Diagnosis~Concave_points3+Area1+Texture3+Concave_points2+Smoothness3, data2, model=rpart, predict=mypredict.rpart)$error
error.bagging2.cv <- errorest(Diagnosis~Concave_points3+Area1+Texture3+Concave_points2+Smoothness3, data2, model=bagging)$error
error.randomForest2.cv <- errorest(Diagnosis~Concave_points3+Area1+Texture3+Concave_points2+Smoothness3, data2, model=randomForest)$error

error.lda3.cv <- errorest(Diagnosis~Concave_points3+Radius2+Concavity3+Compactness1, data2, model=lda, predict=my.predict.lqda, estimator="cv", est.para=control.errorest(k = 10))$error
error.qda3.cv <- errorest(Diagnosis~Concave_points3+Radius2+Concavity3+Compactness1, data2, model=qda, predict=my.predict.lqda, estimator="cv", est.para=control.errorest(k = 10))$error
error.knn3.cv <- errorest(Diagnosis~Concave_points3+Radius2+Concavity3+Compactness1, data2, model=my.ipredknn, predict=my.predict.knn, estimator="cv", est.para=control.errorest(k = 10), n.of.neighbors=5)$error
error.logit3.cv <- errorest(Diagnosis~Concave_points3+Radius2+Concavity3+Compactness1, data2, model=my.glm, predict=my.predict.glm, estimator="cv", est.para=control.errorest(k = 10), family1=binomial(link="logit"))$error
error.tree3.cv <- errorest(Diagnosis~Concave_points3+Radius2+Concavity3+Compactness1, data2, model=rpart, predict=mypredict.rpart)$error
error.bagging3.cv <- errorest(Diagnosis~Concave_points3+Radius2+Concavity3+Compactness1, data2, model=bagging)$error
error.randomForest3.cv <- errorest(Diagnosis~Concave_points3+Radius2+Concavity3+Compactness1, data2, model=randomForest)$error

error.lda4.cv <- errorest(Diagnosis ~., data2, model=lda, predict=my.predict.lqda, estimator="cv", est.para=control.errorest(k = 10))$error
error.qda4.cv <- errorest(Diagnosis ~., data2, model=qda, predict=my.predict.lqda, estimator="cv", est.para=control.errorest(k = 10))$error
error.knn4.cv <- errorest(Diagnosis~., data2, model=my.ipredknn, predict=my.predict.knn, estimator="cv", est.para=control.errorest(k = 10), n.of.neighbors=5)$error
error.logit4.cv <- errorest(Diagnosis~., data2, model=my.glm, predict=my.predict.glm, estimator="cv", est.para=control.errorest(k = 10), family1=binomial(link="logit"))$error
error.tree4.cv <- errorest(Diagnosis~., data2, model=rpart, predict=mypredict.rpart)$error
error.bagging4.cv <- errorest(Diagnosis~., data2, model=bagging)$error
error.randomForest4.cv <- errorest(Diagnosis~., data2, model=randomForest)$error

errors.cv <- matrix(c(error.lda1.cv, error.qda1.cv, error.knn1.cv, error.logit1.cv, error.tree1.cv, error.bagging1.cv, error.randomForest1.cv, error.lda2.cv, error.qda2.cv, error.knn2.cv, error.logit2.cv, error.tree2.cv, error.bagging2.cv, error.randomForest2.cv, error.lda3.cv, error.qda3.cv, error.knn3.cv, error.logit3.cv, error.tree3.cv, error.bagging3.cv, error.randomForest3.cv, error.lda4.cv, error.qda4.cv, error.knn4.cv, error.logit4.cv, error.tree4.cv, error.bagging4.cv, error.randomForest4.cv), nrow=7, ncol=4, dimnames =list(c("error.lda.cv", "error.qda.cv", "error.knn.cv",  "error.logit.cv", "error.tree.cv", "error.bagging.cv", "error.randomForest.cv"), c("1 features subset", "2 features subset","3 features subset", "4 features subset")))
accuracy.cv <- (1-errors.cv)*100
rownames(accuracy.cv) <- c("accuracy.lda.cv", "accuracy.qda.cv", "accuracy.knn.cv",  "accuracy.logit.cv", "accuracy.tree.cv", "accuracy.bagging.cv", "accuracy.randomForest.cv")
accuracy.cv <- round(accuracy.cv, 3)

#------------------------------------------------------------------------------------
# Learning and test set for standardized data
#------------------------------------------------------------------------------------

dataTransform <- preProcess(learning.set, method=c("center", "scale"))
learning.set.s <- predict(dataTransform, learning.set)
test.set.s  <- predict(dataTransform, test.set)

data.lda1.s <- lda(Diagnosis~Concave_points3+Area1+Texture3+Radius2+Fractal_dimension2, data=learning.set.s)
data.qda1.s <- qda(Diagnosis~Concave_points3+Area1+Texture3+Radius2+Fractal_dimension2, data=learning.set.s)
data.knn1.s <- ipredknn(Diagnosis~Concave_points3+Area1+Texture3+Radius2+Fractal_dimension2, data=learning.set.s, k=5)
data.logit1.s <- glm(Diagnosis~Concave_points3+Area1+Texture3+Radius2+Fractal_dimension2, data=learning.set.s, family=binomial(link="logit"))
data.tree1.s <- rpart(Diagnosis~Concave_points3+Area1+Texture3+Radius2+Fractal_dimension2, data=learning.set.s)
data.bagging1.s <- bagging(Diagnosis~Concave_points3+Area1+Texture3+Radius2+Fractal_dimension2, data=learning.set.s)
data.randomForest1.s <- randomForest(Diagnosis~Concave_points3+Area1+Texture3+Radius2+Fractal_dimension2, data=learning.set.s)

data.lda2.s <- lda(Diagnosis~Concave_points3+Area1+Texture3+Concave_points2+Smoothness3, data=learning.set.s)
data.qda2.s <- qda(Diagnosis~Concave_points3+Area1+Texture3+Concave_points2+Smoothness3, data=learning.set.s)
data.knn2.s <- ipredknn(Diagnosis~Concave_points3+Area1+Texture3+Concave_points2+Smoothness3, data=learning.set.s, k=5)
data.logit2.s <- glm(Diagnosis~Concave_points3+Area1+Texture3+Concave_points2+Smoothness3, data=learning.set.s, family=binomial(link="logit"))
data.tree2.s <- rpart(Diagnosis~Concave_points3+Area1+Texture3+Concave_points2+Smoothness3, data=learning.set.s)
data.bagging2.s <- bagging(Diagnosis~Concave_points3+Area1+Texture3+Concave_points2+Smoothness3, data=learning.set.s)
data.randomForest2.s <- randomForest(Diagnosis~Concave_points3+Area1+Texture3+Concave_points2+Smoothness3, data=learning.set.s)

data.lda3.s <- lda(Diagnosis~Concave_points3+Radius2+Concavity3+Compactness1, data=learning.set.s)
data.qda3.s <- qda(Diagnosis~Concave_points3+Radius2+Concavity3+Compactness1, data=learning.set.s)
data.knn3.s <- ipredknn(Diagnosis~Concave_points3+Radius2+Concavity3+Compactness1, data=learning.set.s, k=5)
data.logit3.s <- glm(Diagnosis~Concave_points3+Radius2+Concavity3+Compactness1, data=learning.set.s, family=binomial(link="logit"))
data.tree3.s <- rpart(Diagnosis~Concave_points3+Radius2+Concavity3+Compactness1, data=learning.set.s)
data.bagging3.s <- bagging(Diagnosis~Concave_points3+Radius2+Concavity3+Compactness1, data=learning.set.s)
data.randomForest3.s <- randomForest(Diagnosis~Concave_points3+Radius2+Concavity3+Compactness1, data=learning.set.s)

data.lda4.s <- lda(Diagnosis~., data=learning.set.s)
data.qda4.s <- qda(Diagnosis~., data=learning.set.s)
data.knn4.s <- ipredknn(Diagnosis~., data=learning.set.s, k=5)
data.logit4.s <- glm(Diagnosis~., data=learning.set.s, family=binomial(link="logit"))
data.tree4.s <- rpart(Diagnosis~., data=learning.set.s)
data.bagging4.s <- bagging(Diagnosis~., data=learning.set.s)
data.randomForest4.s <- randomForest(Diagnosis~., data=learning.set.s)

prediction.logit1.s  <-  predict(data.logit1.s, test.set.s[,-1])
prediction.logit2.s  <-  predict(data.logit2.s, test.set.s[,-1])
prediction.logit3.s  <-  predict(data.logit3.s, test.set.s[,-1])
prediction.logit4.s  <-  predict(data.logit4.s, test.set.s[,-1])

pred.labels.lda1.s <- predict(data.lda1.s, test.set.s[,-1])$class
pred.labels.qda1.s <- predict(data.qda1.s, test.set.s[,-1])$class
pred.labels.knn1.s <- predict(data.knn1.s, test.set.s[,-1], type="class")
pred.labels.logit1.s <- prob2labels(probs=prediction.logit1.s, cutoff=0.5)
pred.labels.tree1.s <- predict(data.tree1.s, test.set.s[,-1], type="class")
pred.labels.bagging1.s <- predict(data.bagging1.s, test.set.s[,-1], type="class")
pred.labels.randomForest1.s <- predict(data.randomForest1.s, test.set.s[,-1], type="class")

pred.labels.lda2.s <- predict(data.lda2.s, test.set.s[,-1])$class
pred.labels.qda2.s <- predict(data.qda2.s, test.set.s[,-1])$class
pred.labels.knn2.s <- predict(data.knn2.s, test.set.s[,-1], type="class")
pred.labels.logit2.s <- prob2labels(probs=prediction.logit2.s, cutoff=0.5)
pred.labels.tree2.s <- predict(data.tree2.s, test.set.s[,-1], type="class")
pred.labels.bagging2.s <- predict(data.bagging2.s, test.set.s[,-1], type="class")
pred.labels.randomForest2.s <- predict(data.randomForest2.s, test.set.s[,-1], type="class")

pred.labels.lda3.s <- predict(data.lda3.s, test.set.s[,-1])$class
pred.labels.qda3.s <- predict(data.qda3.s, test.set.s[,-1])$class
pred.labels.knn3.s <- predict(data.knn3.s, test.set.s[,-1], type="class")
pred.labels.logit3.s <- prob2labels(probs=prediction.logit3.s, cutoff=0.5)
pred.labels.tree3.s <- predict(data.tree3.s, test.set.s[,-1], type="class")
pred.labels.bagging3.s <- predict(data.bagging3.s, test.set.s[,-1], type="class")
pred.labels.randomForest3.s <- predict(data.randomForest3.s, test.set.s[,-1], type="class")

pred.labels.lda4.s <- predict(data.lda4.s, test.set.s[,-1])$class
pred.labels.qda4.s <- predict(data.qda4.s, test.set.s[,-1])$class
pred.labels.knn4.s <- predict(data.knn4.s, test.set.s[,-1], type="class")
pred.labels.logit4.s <- prob2labels(probs=prediction.logit4.s, cutoff=0.5)
pred.labels.tree4.s <- predict(data.tree4.s, test.set.s[,-1], type="class")
pred.labels.bagging4.s <- predict(data.bagging4.s, test.set.s[,-1], type="class")
pred.labels.randomForest4.s <- predict(data.randomForest4.s, test.set.s[,-1], type="class")

conf.mat.lda1.s <- table(pred.labels.lda1.s, real.labels) 
conf.mat.qda1.s <- table(pred.labels.qda1.s, real.labels)
conf.mat.knn1.s <- table(pred.labels.knn1.s, real.labels)
conf.mat.logit1.s <- table(pred.labels.logit1.s, real.labels)
conf.mat.tree1.s <- table(pred.labels.tree1.s, real.labels)
conf.mat.bagging1.s <- table(pred.labels.bagging1.s, real.labels)
conf.mat.randomForest1.s <- table(pred.labels.randomForest1.s, real.labels)

conf.mat.lda2.s <- table(pred.labels.lda2.s, real.labels) 
conf.mat.qda2.s <- table(pred.labels.qda2.s, real.labels)
conf.mat.knn2.s <- table(pred.labels.knn2.s, real.labels)
conf.mat.logit2.s <- table(pred.labels.logit2.s, real.labels)
conf.mat.tree2.s <- table(pred.labels.tree2.s, real.labels)
conf.mat.bagging2.s <- table(pred.labels.bagging2.s, real.labels)
conf.mat.randomForest2.s <- table(pred.labels.randomForest2.s, real.labels)

conf.mat.lda3.s <- table(pred.labels.lda3.s, real.labels) 
conf.mat.qda3.s <- table(pred.labels.qda3.s, real.labels)
conf.mat.knn3.s <- table(pred.labels.knn3.s, real.labels)
conf.mat.logit3.s <- table(pred.labels.logit3.s, real.labels)
conf.mat.tree3.s <- table(pred.labels.tree3.s, real.labels)
conf.mat.bagging3.s <- table(pred.labels.bagging3.s, real.labels)
conf.mat.randomForest3.s <- table(pred.labels.randomForest3.s, real.labels)

conf.mat.lda4.s <- table(pred.labels.lda4.s, real.labels) 
conf.mat.qda4.s <- table(pred.labels.qda4.s, real.labels)
conf.mat.knn4.s <- table(pred.labels.knn4.s, real.labels)
conf.mat.logit4.s <- table(pred.labels.logit4.s, real.labels)
conf.mat.tree4.s <- table(pred.labels.tree4.s, real.labels)
conf.mat.bagging4.s <- table(pred.labels.bagging4.s, real.labels)
conf.mat.randomForest4.s <- table(pred.labels.randomForest4.s, real.labels)

error.lda1.s <- (n.test-sum(diag(conf.mat.lda1.s)))/n.test
error.qda1.s <- (n.test-sum(diag(conf.mat.qda1.s)))/n.test
error.knn1.s <- (n.test-sum(diag(conf.mat.knn1.s)))/n.test
error.logit1.s <- (n.test-sum(diag(conf.mat.logit1.s)))/n.test
error.tree1.s <- (n.test-sum(diag(conf.mat.tree1.s)))/n.test
error.bagging1.s <- (n.test-sum(diag(conf.mat.bagging1.s)))/n.test
error.randomForest1.s <- (n.test-sum(diag(conf.mat.randomForest1.s)))/n.test

error.lda2.s <- (n.test-sum(diag(conf.mat.lda2.s)))/n.test
error.qda2.s <- (n.test-sum(diag(conf.mat.qda2.s)))/n.test
error.knn2.s <- (n.test-sum(diag(conf.mat.knn2.s)))/n.test
error.logit2.s <- (n.test-sum(diag(conf.mat.logit2.s)))/n.test
error.tree2.s <- (n.test-sum(diag(conf.mat.tree2.s)))/n.test
error.bagging2.s<- (n.test-sum(diag(conf.mat.bagging2.s)))/n.test
error.randomForest2.s <- (n.test-sum(diag(conf.mat.randomForest2.s)))/n.test

error.lda3.s <- (n.test-sum(diag(conf.mat.lda3.s)))/n.test
error.qda3.s <- (n.test-sum(diag(conf.mat.qda3.s)))/n.test
error.knn3.s <- (n.test-sum(diag(conf.mat.knn3.s)))/n.test
error.logit3.s <- (n.test-sum(diag(conf.mat.logit3.s)))/n.test 
error.tree3.s <- (n.test-sum(diag(conf.mat.tree3.s)))/n.test
error.bagging3.s <- (n.test-sum(diag(conf.mat.bagging3.s)))/n.test
error.randomForest3.s <- (n.test-sum(diag(conf.mat.randomForest3.s)))/n.test

error.lda4.s <- (n.test-sum(diag(conf.mat.lda4.s)))/n.test
error.qda4.s <- (n.test-sum(diag(conf.mat.qda4.s)))/n.test
error.knn4.s <- (n.test-sum(diag(conf.mat.knn4.s)))/n.test
error.logit4.s <- (n.test-sum(diag(conf.mat.logit4.s)))/n.test
error.tree4.s <- (n.test-sum(diag(conf.mat.tree4.s)))/n.test
error.bagging4.s <- (n.test-sum(diag(conf.mat.bagging4.s)))/n.test
error.randomForest4.s <- (n.test-sum(diag(conf.mat.randomForest4.s)))/n.test

errors.s <- matrix(c(error.lda1.s, error.qda1.s, error.knn1.s, error.logit1.s, error.tree1.s, error.bagging1.s, error.randomForest1.s, error.lda2.s, error.qda2.s, error.knn2.s, error.logit2.s, error.tree2.s, error.bagging2.s, error.randomForest2.s, error.lda3.s, error.qda3.s, error.knn3.s, error.logit3.s, error.tree3.s, error.bagging3.s, error.randomForest3.s, error.lda4.s, error.qda4.s, error.knn4.s, error.logit4.s, error.tree4.s, error.bagging4.s, error.randomForest4.s), nrow=7, ncol=4, dimnames =list(c("error.lda", "error.qda", "error.knn",  "error.logit", "error.tree", "error.bagging", "error.randomForest"), c("1 features subset", "2 features subset","3 features subset", "4 features subset")))
errors.s <- round(errors.s, 5)
accuracy.s <- (1-errors.s)*100
rownames(accuracy.s) <- c("accuracy.lda", "accuracy.qda", "accuracy.knn",  "accuracy.logit", "accuracy.tree", "accuracy.bagging", "accuracy.randomForest")


# accuracy difference between classifiers based on non-standardized and standardized data
accuracy.ns.s <- round(accuracy - accuracy.s, 5)
