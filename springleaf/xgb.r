library(readr)
library(xgboost)

set.seed(721) 

cat("reading the train and test data\n")

###########
# real test
###########
train <- read_csv("./data/train.csv")
test  <- read_csv("./data/test.csv")
test$target = 2

######################
# for cross validation
######################
#train <- read_csv("./data/tr.csv")
#test  <- read_csv("./data/cv.csv")

all <- rbind(train,test)
target_id_columns = c("target","ID")
all_target_id <- all[,target_id_columns]
other_columns <- setdiff(names(all),target_id_columns)

###################################################################
###################################################################
# DATA PREPROCESSING
###################################################################
###################################################################

# this portion removing variables is borrowed from: https://www.kaggle.com/raddar/springleaf-marketing-response/removing-irrelevant-vars

# train.unique.count=lapply(train, function(x) length(unique(x)))
# train.unique.count_1=unlist(train.unique.count[unlist(train.unique.count)==1])
# train.unique.count_2=unlist(train.unique.count[unlist(train.unique.count)==2])
# train.unique.count_2=train.unique.count_2[-which(names(train.unique.count_2)=='target')]
# 
# delete_const=names(train.unique.count_1)
# 
# delete_NA56=names(which(unlist(lapply(train[,(names(train) %in% names(train.unique.count_2))], function(x) max(table(x,useNA='always'))))==145175))
# delete_NA89=names(which(unlist(lapply(train[,(names(train) %in% names(train.unique.count_2))], function(x) max(table(x,useNA='always'))))==145142))
# delete_NA918=names(which(unlist(lapply(train[,(names(train) %in% names(train.unique.count_2))], function(x) max(table(x,useNA='always'))))==144313))


##########################
# CONSTANT features : 
# drop them
##########################

delete_const = c("VAR_0205","VAR_0207","VAR_0213","VAR_0214","VAR_0840","VAR_0847","VAR_1428")
delete_NA = c("VAR_0205","VAR_0207","VAR_0213","VAR_0214","VAR_0840","VAR_0847","VAR_1428","VAR_0008","VAR_0009","VAR_0010","VAR_0011","VAR_0012","VAR_0018","VAR_0019","VAR_0020","VAR_0021","VAR_0022","VAR_0023","VAR_0024","VAR_0025","VAR_0026","VAR_0027","VAR_0028","VAR_0029","VAR_0030","VAR_0031","VAR_0032","VAR_0038","VAR_0039","VAR_0040","VAR_0041","VAR_0042","VAR_0043","VAR_0044","VAR_0196","VAR_0197","VAR_0199","VAR_0202","VAR_0203","VAR_0215","VAR_0216","VAR_0221","VAR_0222","VAR_0223","VAR_0229","VAR_0239","VAR_0188","VAR_0189","VAR_0190","VAR_0246","VAR_0394","VAR_0438","VAR_0446","VAR_0527","VAR_0528","VAR_0530")
delete_ID = c("VAR_0212","VAR_0227","VAR_0228")
delete_columns = c(delete_const,delete_NA,delete_ID)
all <- all[,!(names(all) %in% delete_columns)]
other_columns <- setdiff(other_columns,delete_columns)
save(delete_columns,file='./data/delete_columns.Rda')

#############################
# CATEGORY features : 
# one hot encode them into binary
##############################

train <- train[,!(names(train) %in% delete_columns)]
train_count <- lapply(train, function(x) length(unique(x)))
all_count <- lapply(all, function(x) length(unique(x)))
M = cbind(as.integer(train_count),as.integer(all_count))
category_candidate = ((M[,1] == M[,2]) & M[,2]<15)
hist(M[category_candidate,2],breaks=200,xlim=c(0,200))
category_columns = names(train_count[category_candidate])
category_char_columns = c('VAR_0001','VAR_0005','VAR_0226','VAR_0230','VAR_0232','VAR_0236','VAR_0237','VAR_0274','VAR_0283','VAR_0305','VAR_0325','VAR_0342','VAR_0352','VAR_0353','VAR_0354','VAR_0466','VAR_0467','VAR_1934')
category_columns <- union(category_char_columns,category_columns)

all_category <- all[,category_columns]
all_category[is.na(all_category)] <- 123456

# category : from integer to factors
for (f in names(all_category))
{
    print(f)
    if (class(all[[f]])=="character") {
        levels <- unique(all_category[[f]])
        all_category[[f]] <- as.integer(factor(all_category[[f]], levels=levels))
    }
    all_category[[f]] <- factor(all_category[[f]])
}
# category : one hot encoding factors into binary
all_binary <- data.frame(model.matrix(~.-1,all_category))
gc()
save(all_binary,file='./data/all_binary.Rda')
other_columns <- setdiff(other_columns,category_columns)
save(category_columns,file='./data/category_columns.Rda')

######################
# CHARACTER feature :
# let's feature engineer some
# TODO
# TODO
# TODO
# TODO
#######################

load(file='./data/category_columns.Rda')
char_columns <- names(all[1:1000,lapply(all, function(x) class(x)) == "character"])
char_columns <- setdiff(char_columns,category_columns)
save(char_columns,file='./data/char_columns.Rda')

# char_columns <- c("VAR_0073", "VAR_0075", "VAR_0156", "VAR_0157", "VAR_0158", "VAR_0159", "VAR_0166", "VAR_0167", "VAR_0168", "VAR_0169", "VAR_0176", "VAR_0177", "VAR_0178", "VAR_0179", "VAR_0200", "VAR_0204", "VAR_0217", "VAR_0404", "VAR_0493")
# rbind(train[1:10,char_columns],train_count[char_columns])

##########################
# CONTINUOUS features : 
# normalize them
##########################

load(file='./data/delete_columns.Rda')
load(file='./data/char_columns.Rda')
load(file='./data/category_columns.Rda')
continuous_columns <- setdiff(names(all),c(category_columns,char_columns,target_id_columns,delete_columns))
all_continuous <- all[,continuous_columns]
replace_set = c(999999994,999999995,999999996,999999997,999999998,999999999,-99999,999999,9999,9998,9994,9995,9996,9997,9998,999,998,997,996,995,994,99,-1)

for (f in names(all_continuous))
{
    v = all_continuous[[f]]
    v[v %in% replace_set] <- NA
    v[is.na(v)] = mean(v, na.rm = TRUE)
    all_continuous[[f]] <- scale(v)
    print(c(f,min(v),max(v)))
}

save(all_continuous,file='./data/all_continuous.Rda')
save(continuous_columns,file='./data/continuous_columns.Rda')

##########################
##########################
##########################
##########################
# JOIN EVERY THING ....
##########################
##########################
##########################

load(file='./data/all_continuous.Rda')
load(file='./data/all_target_id.Rda')
load(file='./data/all_binary.Rda')

all <- cbind(all_target_id,all_binary,all_continuous)

train <- all[all$target<2,]
test  <- all[all$target==2,]

save(train,file='./data/train.Rda')
save(test,file='./data/test.Rda')

###################################################################
###################################################################
# XGB TRAINING
###################################################################
###################################################################

library(readr)
library(xgboost)

set.seed(15111975) # alternative one...

load(file='./data/train.Rda')

feature.names = setdiff(names(train),c("target","ID"))

h <- sample(nrow(train), as.integer(nrow(train)*0.9))

val<-train[-h,]
gc()

train <-train[h,]
gc()

dtrain <- xgb.DMatrix(data.matrix(train[,feature.names]), label=train$target)

train=train[1:3,]
gc()


dval <- xgb.DMatrix(data.matrix(val[,feature.names]), label=val$target)
val=val[1:3,]
gc()

watchlist <- watchlist <- list(eval = dval, train = dtrain)

# this one leads to 0.078975
# (eval-auc:0.794878	train-auc:0.999710) 800 nrounds (xgb_12.csv) :
# param <- list(  objective           = "binary:logistic", 
#                 # booster = "gblinear",
#                 eta                 = 0.025, #0.06, #0.01,
#                 max_depth           = 11,  # changed from default of 8
#                 subsample           = 0.7,
#                 colsample_bytree    = 0.7,
#                 eval_metric         = "auc"
#                 # alpha = 0.0001, 
#                 # lambda = 1
#                 )
# 
# clf <- xgb.train(   params              = param, 
#                     data                = dtrain, 
#                     nrounds             = 800,
#                     verbose             = 2, 
#                     early.stop.round    = NULL,
#                     watchlist           = watchlist,
#                     maximize            = TRUE)


# (eval-auc:0.792568	train-auc:0.925411) 800 nrounds : 0.79248
# [2999]	eval-auc:0.796404	train-auc:0.990794 (xgb_nikko_14.csv) : ??? 
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
#                     nrounds             = 3000,
#                     verbose             = 2, 
#                     early.stop.round    = NULL,
#                     watchlist           = watchlist,
#                     maximize            = TRUE)

# [800]     eval-auc:0.794843	train-auc:0.992240 : 0.79452
# [1358]    eval-auc:0.796350	train-auc:0.999144 : ??????? (xgb_nikko_15.csv)

param <- list(  objective           = "binary:logistic", 
                # booster = "gblinear",
                eta                 = 0.025,
                max_depth           = 11,
                subsample           = 0.7,
                colsample_bytree    = 0.7,
                min_child_weight    = 7,
                gamma               = 2,
                eval_metric         = "auc"
                # alpha = 0.0001, 
                # lambda = 1
                )

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 1400,
                    verbose             = 2, 
                    early.stop.round    = NULL,
                    watchlist           = watchlist,
                    maximize            = TRUE)


# (eval-auc:0.793176	train-auc:0.959704) 800 nrounds : 0.79330
# [2999]	eval-auc:0.793908	train-auc:0.999801 (xgb_nikko_13.csv) : ??? 
# param <- list(  objective           = "binary:logistic", 
#                 # booster = "gblinear",
#                 eta                 = 0.02, #0.06, #0.01,
#                 max_depth           = 10,  # changed from default of 8
#                 subsample           = 0.5,
#                 colsample_bytree    = 0.5,
#                 min_child_weight    = 7,
#                 gamma               = 1,
#                 eval_metric         = "auc",
#                 # alpha = 0.0001, 
#                 lambda = 1
#                 )
# 
# clf <- xgb.train(   params              = param, 
#                     data                = dtrain, 
#                     nrounds             = 3000,
#                     verbose             = 2, 
#                     early.stop.round    = NULL,
#                     watchlist           = watchlist,
#                     maximize            = TRUE)
# 




dtrain=0
gc()

dval=0
gc()

load(file='./data/test.Rda')

submission <- data.frame(ID=test$ID)
submission$target <- NA 
for (rows in split(1:nrow(test), ceiling((1:nrow(test))/10000))) 
{
    print(max(rows))
    submission[rows, "target"] <- predict(clf, data.matrix(test[rows,feature.names]))
}

cat("saving the test submission file\n")
write_csv(submission, "./data/xgb_test.csv")

