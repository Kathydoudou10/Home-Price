library(Sleuth2)
library(glmnet)
library(caret)
library(leaps)
library(bestglm)
library(AppliedPredictiveModeling)
library(e1071)
library(caret)
library(corrplot)
library(ISLR)
library(Matrix)
library(foreach)
library(lattice)
library(ggplot2)
library(VIM)
library(scales)
library(gridExtra)
library(plyr)



train <- read.csv('Desktop/lessons/6302/Assignment1/train.csv',stringsAsFactors = F)
test <- read.csv('Desktop/lessons/6302/Assignment1/test.csv',stringsAsFactors = F)


###################
## data cleaning ##
###################

##combine train and test into one file
test_labels <- test$Id
test$Id <- NULL
train$Id <- NULL
test$SalePrice <- NA
all <- rbind(train, test) 
dim(all)
##check duplicated data
duplicated(all)
##removing outliers
all <- all[-c(524, 1299),] 
##check NA value
colSums(is.na(all))
##change some variables NA to avoid misunderstand
all$PoolQC[is.na(all$PoolQC)] <- 'None'
Qualities <- c('None' = 0, 'Po' = 1, 'Fa' = 2, 'TA' = 3, 'Gd' = 4, 'Ex' = 5)
all$PoolQC<-as.integer(revalue(all$PoolQC, Qualities))

all$MiscFeature[is.na(all$MiscFeature)] <- 'None'
all$Alley[is.na(all$Alley)] <- 'None'
all$Fence[is.na(all$Fence)] <- 'None'

all$FireplaceQu[is.na(all$FireplaceQu)] <- 'None'
all$FireplaceQu<-as.integer(revalue(all$FireplaceQu, Qualities))

all$GarageType[is.na(all$GarageType)] <- 'None'

all$GarageFinish[is.na(all$GarageFinish)] <- 'None'
Finish <- c('None'=0, 'Unf'=1, 'RFn'=2, 'Fin'=3)
all$GarageFinish<-as.integer(revalue(all$GarageFinish, Finish))

all$GarageQual[is.na(all$GarageQual)] <- 'None'
all$GarageQual<-as.integer(revalue(all$GarageQual, Qualities))

all$GarageCond[is.na(all$GarageCond)] <- 'None'
all$GarageCond<-as.integer(revalue(all$GarageCond, Qualities))

all$BsmtQual[is.na(all$BsmtQual)] <- 'None'
all$BsmtQual<-as.integer(revalue(all$BsmtQual, Qualities))

all$BsmtCond[is.na(all$BsmtCond)] <- 'None'
all$BsmtCond<-as.integer(revalue(all$BsmtCond, Qualities))

all$BsmtExposure[is.na(all$BsmtExposure)] <- 'None'
Exposure <- c('None'=0, 'No'=1, 'Mn'=2, 'Av'=3, 'Gd'=4)
all$BsmtExposure<-as.integer(revalue(all$BsmtExposure, Exposure))

all$BsmtFinType1[is.na(all$BsmtFinType1)] <- 'None'
FinType <- c('None'=0, 'Unf'=1, 'LwQ'=2, 'Rec'=3, 'BLQ'=4, 'ALQ'=5, 'GLQ'=6)
all$BsmtFinType1<-as.integer(revalue(all$BsmtFinType1, FinType))

all$BsmtFinType2[is.na(all$BsmtFinType2)] <- 'None'
FinType <- c('None'=0, 'Unf'=1, 'LwQ'=2, 'Rec'=3, 'BLQ'=4, 'ALQ'=5, 'GLQ'=6)
all$BsmtFinType2<-as.integer(revalue(all$BsmtFinType2, FinType))


##delete "Utilities"
all <- all[ , !(colnames(all) %in% c("Utilities") ) ]
##change some variables into Character
all$YrSold <- as.character(all$YrSold)
all$MoSold <- as.character(all$MoSold)
all$MSSubClass <- as.character(all$MSSubClass)


#########################
## Feature engineering ##
#########################

##count total number of bathrooms
all$TotBathrooms <- all$FullBath + (all$HalfBath*0.5) + all$BsmtFullBath + (all$BsmtHalfBath*0.5)

##check if there has been no Remodeling/Addition, add "Age", "Remod" variables.
all$Remod <- ifelse(all$YearBuilt==all$YearRemodAdd, 0, 1) 
#0=No Remodeling, 1=Remodeling 
all$Age <- as.numeric(all$YrSold)-all$YearRemodAdd

##add IsNew variable to know if the house is sold the year it built.
all$IsNew <- ifelse(all$YrSold==all$YearBuilt, 1, 0) 
table(all$IsNew)

##adds up the living space above and below ground.
all$TotalSqFeet <- all$GrLivArea + all$TotalBsmtSF

## taking the log for all numeric predictors with an absolute skew greater than 0.8.
##(actually: log+1, to avoid division by zero issues).
numericVars <- which(sapply(all, is.numeric))
numericVarNames <- names(numericVars)
numericVarNames <- numericVarNames[!(numericVarNames %in% c('MSSubClass', 
                                                            'MoSold', 'YrSold', 'SalePrice', 'OverallQual', 'OverallCond'))] 
#numericVarNames was created before having done anything 
numericVarNames <- append(numericVarNames, c('Age', 'TotalPorchSF',
                                             'TotBathrooms', 'TotalSqFeet')) 
DFnumeric <- all[, names(all) %in% numericVarNames]

for(i in 1:ncol(DFnumeric)){ if(abs(skewness(DFnumeric[,i],na.rm = TRUE))>0.8){ 
  DFnumeric[,i] <- log(DFnumeric[,i] +1) }
}

##Normalizing the data
PreNum <- preProcess(DFnumeric, method=c("center", "scale")) 
print(PreNum)
DFnorm <- predict(PreNum, DFnumeric) 
dim(DFnorm)


##################
##Filtering Data##
##################

##Identify and remove predictors with near-zero variance
DFcharacter <- all[, !names(all) %in% numericVarNames]
nearZeroVar(DFcharacter)
colnames(DFcharacter)[nearZeroVar(DFcharacter)]
DFcharacter.1 <- DFcharacter[,-nearZeroVar(DFcharacter)]
nearZeroVar(DFnumeric)
colnames(DFnumeric)[nearZeroVar(DFnumeric)]
DFnumeric.1 <- DFnumeric[,-nearZeroVar(DFnumeric)]
##check
dim(DFcharacter.1)
dim(DFnumeric.1)

##Identify and remove highly correlated predictors
DFnumericCorr <- cor(DFnumeric.1, use="pairwise.complete.obs")
apply(DFnumericCorr,1 ,function(x) abs(x)>0.75)
DFnumeric.2 <- DFnumeric.1[, !(colnames(DFnumeric.1) %in% 
                                 c('X1stFlrSF','LotFrontage','GarageYrBlt','TotRmsAbvGrd','GarageCars'))]
all.1 <- cbind(DFnumeric.2, DFcharacter.1) 
##check
dim(all.1)
str(all.1)

##Composing train and test sets and logSalePrice
train1 <- all.1[!is.na(all.1$SalePrice),]
train1$SalePrice <- log(train1$SalePrice)
test1 <- all.1[is.na(all.1$SalePrice),]
test1$SalePrice <- 0
all.2 <- rbind(train1, test1) 
#check
colSums(is.na(all.2))


##imputation
all.2$MSZoning[is.na(all.2$MSZoning)] <- names(sort(-table(all.2$MSZoning)))[1]
all.2$Exterior1st[is.na(all.2$Exterior1st)] <- names(sort(-table(all.2$Exterior1st)))[1]
all.2$Exterior2nd[is.na(all.2$Exterior2nd)] <- names(sort(-table(all.2$Exterior2nd)))[1]
all.2$MasVnrType[is.na(all.2$MasVnrType)] <- names(sort(-table(all.2$MasVnrType)))[1]
all.2$Electrical[is.na(all.2$Electrical)] <- names(sort(-table(all.2$Electrical)))[1]
all.2$KitchenQual[is.na(all.2$KitchenQual)] <- 'TA' #replace with most common value
all.2$KitchenQual<-as.integer(revalue(all.2$KitchenQual, Qualities))
all.2$SaleType[is.na(all.2$SaleType)] <- names(sort(-table(all.2$SaleType)))[1]


all.2$MasVnrArea[is.na(all.2$MasVnrArea)] <-0
all.2$BsmtFinSF1[is.na(all.2$BsmtFinSF1)] <-0
all.2$BsmtUnfSF[is.na(all.2$BsmtUnfSF)] <-0
all.2$TotalBsmtSF[is.na(all.2$TotalBsmtSF)] <-0
all.2$BsmtFullBath[is.na(all.2$BsmtFullBath)] <-0
all.2$BsmtHalfBath[is.na(all.2$BsmtHalfBath)] <-0
all.2$GarageArea[is.na(all.2$GarageArea)] <-0
all.2$TotBathrooms[is.na(all.2$TotBathrooms)] <-0
all.2$TotalSqFeet[is.na(all.2$TotalSqFeet)] <-0


#check
colSums(is.na(all.2))
str(all.2)

##modeling
matrix1 <-model.matrix(SalePrice~.,all.2)
x = matrix1[1:1458,]
y <- all.2$SalePrice[1:1458]
str(matrix1)

####################
##Ridge Regression##
####################

ridge.mod <- cv.glmnet(x, y, alpha=0)
plot(ridge.mod)

##Identify optimal tuning parameter
bestlamR <- ridge.mod$lambda.min
bestlamR
coef(ridge.mod, s=bestlamR)

##Re-fit the model with the optimal lambda and check MSE
ridge.pred <- predict(ridge.mod,x,s=bestlamR)
ridge.rmse <-sqrt(mean(ridge.pred - y)^2)


#############
##The Lasso##
#############

lasso.mod <- cv.glmnet(x, y, alpha=1)
plot(lasso.mod)

##Identify optimal lambda
bestlamL <- lasso.mod$lambda.min
bestlamL
coef(lasso.mod,s=bestlamL)

##Re-fit the model with the optimal lambda and check MSE
lasso.pred <- predict(lasso.mod,x,s=bestlamL)
lasso.rmse <-sqrt(mean(lasso.pred - y)^2)


###############
##Elastic Net##
###############

##Specify that we want repeated 10-fold CV
tcontrol <- trainControl(method="repeatedcv", number=10, repeats=5)

##Define the grid of alpha and lambda values to check
##Cut down on the length and spread of the lambda values
tuneParam <- expand.grid(alpha = seq(0.1, 1, 0.1), lambda = 10^seq(2, -2, length=25))

elasticnet.mod <- train(x, y, trControl=tcontrol, method="glmnet", tuneGrid=tuneParam)
plot(elasticnet.mod)
attributes(elasticnet.mod)
elasticnet.mod$results

##Optimal tuning parameters
elasticnet.mod$bestTune
en.final <- elasticnet.mod$finalModel
coef(en.final, alpha=elasticnet.mod$bestTune$alpha, s=elasticnet.mod$bestTune$lambda)

##Re-fit the model with the optimal lambda and check MSE
elasticnet.pred <- predict(en.final,x,s=elasticnet.mod$bestTune$lambda)
elasticnet.rmse <-sqrt(mean(elasticnet.pred - y)^2)


##############
##Prediction##
##############

x1 = matrix1[1459:2917,]
y <- all.2$SalePrice[1:1458]
##Since Lasso is the optimal,we use exp() to transfer our result
ridge.pred1 <- predict(ridge.mod,x1,s=bestlamR)
ridge.result <- exp(ridge.pred1)
write.csv(ridge.result,file = 'Desktop/lessons/6302/Assignment1/ridge.csv')

lasso.pred1 <- predict(lasso.mod,x1,s=bestlamL)
lasso.result <- exp(lasso.pred1)
write.csv(lasso.result,file = 'Desktop/lessons/6302/Assignment1/lasso.csv')

elasticnet.pred1 <- predict(en.final,x1,s=elasticnet.mod$bestTune$lambda)
elasticnet.result <- exp(elasticnet.pred1)
write.csv(elasticnet.result,file = 'Desktop/lessons/6302/Assignment1/elasticnet.csv')
