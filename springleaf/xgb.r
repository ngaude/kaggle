library(readr)
library(xgboost)

set.seed(721) #seed bag1:8, then eta=0.06not0.04&nround125not250: bag2:64, bag3:6, bag4:88, bag5: 0.03-300-seed666
#bag6:16, train[1:80000,], val=train[80001:120000,], 0.06, 125  #bag7: 888,train[65000:145000,], val=train[1:40000,], 0.06, 125 
#bag8: 888,train[65000:145000,], val=train[1:40000,], 0.03, 300

#seed bag9:9999, 0.02,300,random

#bag10:425, bag11:718, bag12:719, bag13:720, bag14:721


cat("reading the train and test data\n")
#train <- read_csv("./data/train.csv")
#test  <- read_csv("./data/test.csv")
train <- read_csv("./data/tr.csv")
test  <- read_csv("./data/test.csv")
cv <- read_csv("./data/cv.csv")



# this portion removing variables is borrowed from: https://www.kaggle.com/raddar/springleaf-marketing-response/removing-irrelevant-vars

train.unique.count=lapply(train, function(x) length(unique(x)))
train.unique.count_1=unlist(train.unique.count[unlist(train.unique.count)==1])
train.unique.count_2=unlist(train.unique.count[unlist(train.unique.count)==2])
train.unique.count_2=train.unique.count_2[-which(names(train.unique.count_2)=='target')]

delete_const=names(train.unique.count_1)
delete_NA56=names(which(unlist(lapply(train[,(names(train) %in% names(train.unique.count_2))], function(x) max(table(x,useNA='always'))))==145175))
delete_NA89=names(which(unlist(lapply(train[,(names(train) %in% names(train.unique.count_2))], function(x) max(table(x,useNA='always'))))==145142))
delete_NA918=names(which(unlist(lapply(train[,(names(train) %in% names(train.unique.count_2))], function(x) max(table(x,useNA='always'))))==144313))

#VARS to delete
#safe to remove VARS with 56, 89 and 918 NA's as they are covered by other VARS
print(length(c(delete_const,delete_NA56,delete_NA89,delete_NA918)))

train=train[,!(names(train) %in% c(delete_const,delete_NA56,delete_NA89,delete_NA918))]
test=test[,!(names(test) %in% c(delete_const,delete_NA56,delete_NA89,delete_NA918))]
cv=cv[,!(names(cv) %in% c(delete_const,delete_NA56,delete_NA89,delete_NA918))]


print(dim(train))
print(dim(test))
print(dim(cv))

n = dim(train)[1]

feature.names <- names(train)[2:ncol(train)-1]
# names(train)  # 1934 variables

for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]], cv[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
    cv[[f]]  <- as.integer(factor(cv[[f]],  levels=levels))
  }
}

cat("replacing missing values with -1\n")
train[is.na(train)] <- -1
test[is.na(test)]   <- -1
cv[is.na(cv)]   <- -1

h <- sample(nrow(train), as.integer(n*0.8))

val<-train[-h,]
gc()

train <-train[h,]
gc()

#for bag 6
#val=train[80001:120000,]
#train=train[1:80000,]
#gc()

#for bag 7
#val=train[1:40000,]
#train=train[65000:145000,]
#gc()

dtrain <- xgb.DMatrix(data.matrix(train[,feature.names]), label=train$target)

train=train[1:3,]
gc()


dval <- xgb.DMatrix(data.matrix(val[,feature.names]), label=val$target)
val=val[1:3,]
gc()

watchlist <- watchlist <- list(eval = dval, train = dtrain)

# this one leads to 0.078975

param <- list(  objective           = "binary:logistic", 
                # booster = "gblinear",
                eta                 = 0.025, #0.06, #0.01,
                max_depth           = 11,  # changed from default of 8
                subsample           = 0.7,
                colsample_bytree    = 0.7,
                eval_metric         = "auc"
                # alpha = 0.0001, 
                # lambda = 1
                )

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 366, #300, #280, #125, #250, # changed from 300
                    verbose             = 2, 
                    early.stop.round    = 5,
                    watchlist           = watchlist,
                    maximize            = TRUE)


# this one leads to 0.79037
#
# param <- list(  objective           = "binary:logistic", 
#                 # booster = "gblinear",
#                 eta                 = 0.015, #0.06, #0.01,
#                 max_depth           = 9,  # changed from default of 8
#                 subsample           = 0.7,
#                 colsample_bytree    = 0.6,
#                 min_child_weight    = 10,
#                 max_delta_step      = 2,
#                 gamma               = 3,
#                 eval_metric         = "auc"
#                 # alpha = 0.0001, 
#                 # lambda = 1
#                 )
# 
# clf <- xgb.train(   params              = param, 
#                     data                = dtrain, 
#                     nrounds             = 800, #280, #125, #250, # changed from 300
#                     verbose             = 2, 
#                     early.stop.round    = 10,
#                     watchlist           = watchlist,
#                     maximize            = TRUE)

# this one leads to 0.79279
# param <- list(  objective           = "binary:logistic", 
#                 # booster = "gblinear",
#                 eta                 = 0.025, #0.06, #0.01,
#                 max_depth           = 11,  # changed from default of 8
#                 subsample           = 0.7,
#                 colsample_bytree    = 0.7,
#                 min_child_weight    = 7,
#                 gamma               = 2,
#                 eval_metric         = "auc"
#                 # alpha = 0.0001, 
#                 # lambda = 1
#                 )
# 
# clf <- xgb.train(   params              = param, 
#                     data                = dtrain, 
#                     nrounds             = 800, #280, #125, #250, # changed from 300
#                     verbose             = 2, 
#                     early.stop.round    = NULL,
#                     watchlist           = watchlist,
#                     maximize            = TRUE)
# 



dtrain=0
gc()

dval=0
gc()

submission <- data.frame(ID=test$ID)
submission$target <- NA 
for (rows in split(1:nrow(test), ceiling((1:nrow(test))/10000))) {
    submission[rows, "target"] <- predict(clf, data.matrix(test[rows,feature.names]))
 
}
cat("saving the test submission file\n")
write_csv(submission, "./data/xgb_test.csv")

submission <- data.frame(ID=cv$ID)
submission$target <- NA 
for (rows in split(1:nrow(cv), ceiling((1:nrow(cv))/10000))) {
    submission[rows, "target"] <- predict(clf, data.matrix(cv[rows,feature.names]))
 
}
cat("saving the cv submission file\n")
write_csv(submission, "./data/xgb_cv.csv")






