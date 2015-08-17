# load required packages
require(readr)
require(xgboost)
set.seed(20150814)

# read data
xtrain <- read.table(file = "./data/train.csv", header = T, sep = ",")
id_train <- xtrain$ID; xtrain$ID <- NULL
y <- xtrain$target; xtrain$target <- NULL

xtest <- read.table(file = "./data/test.csv", header = T, sep = ",")
id_test <- xtest$ID; xtest$ID <- NULL

xtrain[is.na(xtrain)] <- 9999
xtest[is.na(xtest)] <- 9999

# subset 
class_list <- rep(NA, ncol(xtrain))
for (ii in 1:ncol(xtrain))
{
  class_list[ii] <- class(xtrain[,ii])
}
fact_cols <- which(class_list == "factor") 

# fit model
mod0 <- gbm.fit(x = xtrain[,-fact_cols], y = y, n.trees = 50, 
                shrinkage = 0.01, interaction.depth = 12, distribution = "bernoulli", verbose= T)

# generate prediction
pr <- predict(mod0, xtest[,-fact_cols], mod0$n.trees, type = "response")
pr <- rank(pr)/length(pr)
xfor <- read_csv(file = "./submissions/sample_submission.csv")
xfor$target <- pr

write_csv(xfor, path = "./submissions/btb_20150814.csv")
