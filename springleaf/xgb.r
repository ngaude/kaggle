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

######################
# drop useless columns
######################

delete_const = c("VAR_0205","VAR_0207","VAR_0213","VAR_0214","VAR_0840","VAR_0847","VAR_1428")

delete_NA = c("VAR_0205","VAR_0207","VAR_0213","VAR_0214","VAR_0840","VAR_0847","VAR_1428","VAR_0008","VAR_0009","VAR_0010","VAR_0011","VAR_0012","VAR_0018","VAR_0019","VAR_0020","VAR_0021","VAR_0022","VAR_0023","VAR_0024","VAR_0025","VAR_0026","VAR_0027","VAR_0028","VAR_0029","VAR_0030","VAR_0031","VAR_0032","VAR_0038","VAR_0039","VAR_0040","VAR_0041","VAR_0042","VAR_0043","VAR_0044","VAR_0196","VAR_0197","VAR_0199","VAR_0202","VAR_0203","VAR_0215","VAR_0216","VAR_0221","VAR_0222","VAR_0223","VAR_0229","VAR_0239","VAR_0188","VAR_0189","VAR_0190","VAR_0246","VAR_0394","VAR_0438","VAR_0446","VAR_0527","VAR_0528","VAR_0530")

delete_ID = c("VAR_0212","VAR_0227","VAR_0228")

delete_columns = c(delete_const,delete_NA,delete_ID)

all = rbind(train,test)

all=all[,!(names(all) %in% delete_columns]

# all <- rbind(train,test)
# train_count <- lapply(train, function(x) length(unique(x)))
# all_count <- lapply(all, function(x) length(unique(x)))
# M = cbind(as.integer(train_count),as.integer(all_count))
# categorical_candidate = ((M[,1] == M[,2]) & M[,2]<15)
# hist(M[categorical_candidate,2],breaks=200,xlim=c(0,200))
# names(train_count[categorical_candidate])

######################################
# one hot encoding categorical columns
######################################

categorical_columns_1 = c("VAR_0001","VAR_0005","VAR_0052","VAR_0090","VAR_0091","VAR_0092","VAR_0093","VAR_0098","VAR_0099","VAR_0100","VAR_0101","VAR_0106","VAR_0107","VAR_0108","VAR_0109","VAR_0114","VAR_0115","VAR_0117","VAR_0122","VAR_0123","VAR_0124","VAR_0125","VAR_0130","VAR_0131","VAR_0132","VAR_0138","VAR_0139","VAR_0140","VAR_0141","VAR_0142","VAR_0146","VAR_0150","VAR_0153","VAR_0160","VAR_0170","VAR_0180","VAR_0181","VAR_0182","VAR_0183","VAR_0187","VAR_0191","VAR_0192","VAR_0193","VAR_0194","VAR_0195","VAR_0219","VAR_0220","VAR_0226","VAR_0230","VAR_0232","VAR_0236","VAR_0244","VAR_0245","VAR_0247","VAR_0248","VAR_0249","VAR_0250","VAR_0251","VAR_0252","VAR_0253","VAR_0259","VAR_0260","VAR_0269","VAR_0271","VAR_0275","VAR_0276","VAR_0277","VAR_0278","VAR_0281","VAR_0284","VAR_0285","VAR_0286","VAR_0287","VAR_0290","VAR_0291","VAR_0292","VAR_0305","VAR_0306","VAR_0307","VAR_0308","VAR_0311","VAR_0312","VAR_0325","VAR_0326","VAR_0327","VAR_0328","VAR_0339","VAR_0343","VAR_0344","VAR_0345","VAR_0346","VAR_0347","VAR_0348","VAR_0349","VAR_0350","VAR_0351","VAR_0352","VAR_0353","VAR_0354","VAR_0355","VAR_0356","VAR_0357","VAR_0358","VAR_0362","VAR_0371","VAR_0372","VAR_0377","VAR_0379","VAR_0380","VAR_0383","VAR_0384","VAR_0388","VAR_0389","VAR_0392","VAR_0393","VAR_0395","VAR_0400","VAR_0401","VAR_0402","VAR_0405","VAR_0409","VAR_0416","VAR_0421","VAR_0422","VAR_0430","VAR_0431","VAR_0452","VAR_0457","VAR_0459","VAR_0463","VAR_0466","VAR_0467","VAR_0469","VAR_0470","VAR_0472","VAR_0476","VAR_0477","VAR_0478","VAR_0482","VAR_0483","VAR_0484","VAR_0485","VAR_0486","VAR_0487","VAR_0489","VAR_0490","VAR_0494","VAR_0495","VAR_0496","VAR_0499","VAR_0500","VAR_0502","VAR_0503","VAR_0504","VAR_0505","VAR_0507","VAR_0508","VAR_0513","VAR_0518","VAR_0519","VAR_0520","VAR_0521","VAR_0522","VAR_0523","VAR_0524","VAR_0526","VAR_0529","VAR_0531","VAR_0532","VAR_0538","VAR_0545","VAR_0546","VAR_0547","VAR_0548","VAR_0549","VAR_0563","VAR_0564","VAR_0566","VAR_0567","VAR_0603","VAR_0606","VAR_0621","VAR_0626","VAR_0638","VAR_0639","VAR_0640","VAR_0668","VAR_0678","VAR_0679","VAR_0689","VAR_0703","VAR_0732","VAR_0733","VAR_0735","VAR_0736","VAR_0737","VAR_0739","VAR_0740","VAR_0741","VAR_0745","VAR_0746","VAR_0754","VAR_0761","VAR_0763","VAR_0765","VAR_0773","VAR_0775","VAR_0779","VAR_0784","VAR_0786","VAR_0804","VAR_0923","VAR_0924","VAR_0927","VAR_0928","VAR_0932","VAR_0933","VAR_0941","VAR_0947","VAR_0948","VAR_0953","VAR_0960","VAR_0965","VAR_0967","VAR_0971","VAR_0972","VAR_0994")

categorical_columns_2  = c("VAR_1010","VAR_1012","VAR_1013","VAR_1051","VAR_1055","VAR_1098","VAR_1099","VAR_1100","VAR_1101","VAR_1102","VAR_1103","VAR_1104","VAR_1105","VAR_1106","VAR_1107","VAR_1108","VAR_1109","VAR_1162","VAR_1163","VAR_1164","VAR_1165","VAR_1175","VAR_1176","VAR_1194","VAR_1195","VAR_1196","VAR_1197","VAR_1198","VAR_1205","VAR_1206","VAR_1207","VAR_1213","VAR_1225","VAR_1230","VAR_1239","VAR_1253","VAR_1254","VAR_1256","VAR_1257","VAR_1291","VAR_1305","VAR_1306","VAR_1325","VAR_1326","VAR_1346","VAR_1378","VAR_1379","VAR_1394","VAR_1400","VAR_1401","VAR_1405","VAR_1406","VAR_1407","VAR_1408","VAR_1412","VAR_1413","VAR_1414","VAR_1417","VAR_1422","VAR_1423","VAR_1427","VAR_1429","VAR_1430","VAR_1431","VAR_1432","VAR_1433","VAR_1434","VAR_1435","VAR_1443","VAR_1458","VAR_1505","VAR_1506","VAR_1507","VAR_1508","VAR_1509","VAR_1510","VAR_1511","VAR_1533","VAR_1534","VAR_1538","VAR_1543","VAR_1544","VAR_1547","VAR_1548","VAR_1552","VAR_1553","VAR_1554","VAR_1557","VAR_1558","VAR_1559","VAR_1563","VAR_1586","VAR_1587","VAR_1588","VAR_1590","VAR_1594","VAR_1595","VAR_1632","VAR_1633","VAR_1634","VAR_1635","VAR_1636","VAR_1637","VAR_1638","VAR_1656","VAR_1663","VAR_1671","VAR_1672","VAR_1673","VAR_1674","VAR_1675","VAR_1676","VAR_1678","VAR_1702","VAR_1703","VAR_1704","VAR_1705","VAR_1706","VAR_1707","VAR_1708","VAR_1721","VAR_1722","VAR_1723","VAR_1724","VAR_1725","VAR_1726","VAR_1727","VAR_1728","VAR_1734","VAR_1735","VAR_1746","VAR_1749","VAR_1817","VAR_1818","VAR_1821","VAR_1822","VAR_1844","VAR_1848","VAR_1849","VAR_1862","VAR_1863","VAR_1885","VAR_1896","VAR_1897","VAR_1916","VAR_1917","VAR_1926","VAR_1931","VAR_1934")

#################################/
# dealing with charachter columns
#################################

# char_columns = names(train[1:10,lapply(train, function(x) class(x)) == "character"])
# rbind(train[1:10,char_columns],train_count[char_columns])

char_columns = c("VAR_0001","VAR_0005","VAR_0073","VAR_0075","VAR_0156","VAR_0157","VAR_0158","VAR_0159","VAR_0166","VAR_0167","VAR_0168","VAR_0169","VAR_0176","VAR_0177","VAR_0178","VAR_0179","VAR_0200","VAR_0204","VAR_0217","VAR_0226","VAR_0230","VAR_0232","VAR_0236","VAR_0237","VAR_0274","VAR_0283","VAR_0305","VAR_0325","VAR_0342","VAR_0352","VAR_0353","VAR_0354","VAR_0404","VAR_0466","VAR_0467","VAR_0493","VAR_1934")

categorical_columns_3 = c('VAR_0001','VAR_0005','VAR_0226','VAR_0230','VAR_0232','VAR_0236','VAR_0237','VAR_0274','VAR_0283','VAR_0305','VAR_0325','VAR_0342','VAR_0352','VAR_0353','VAR_0354','VAR_0466','VAR_0467','VAR_1934')


categorical_columns = union(c(categorical_columns_1,categorical_columns_2),categorical_columns_3)
continuous_columns = setdiff(names(all),union(categorical_columns,c("target","ID")))
target_id_columns = c("target","ID")

######################
# character => integer
for (f in names(all)) {
  if (class(all[[f]])=="character") {
    levels <- unique(all[[f]])
    all[[f]] <- as.integer(factor(all[[f]], levels=levels))
  }
}

#####################
# NA => integer
cat("replacing missing values with -1\n")
all[is.na(all)] <- -1

all_category <- all[,categorical_columns]
all_continuous <- all[,continuous_columns]
all_target_id <- all[,target_id_columns]

# categorical integer to factors
for (f in names(all_category))
{
    print(f)
    all_category[[f]] <- factor(all_category[[f]])
}

all_binary <- data.frame(model.matrix(~.-1,all_category))

all <- cbind(all_target_id,all_binary,all_continuous)

train <- all[all$target<2,]
test  <- all[all$target==2,]
save(train,file='./data/train.Rda')
save(test,file='./data/test.Rda')


######################################
######################################
######################################
# HERE HERE HERE
######################################
######################################
######################################

library(readr)
library(xgboost)

set.seed(721) 

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
#                     nrounds             = 366, #300, #280, #125, #250, # changed from 300
#                     verbose             = 2, 
#                     early.stop.round    = 5,
#                     watchlist           = watchlist,
#                     maximize            = TRUE)


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

# this one leads to 0.79351
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

# this one leads to 0.79351
param <- list(  objective           = "binary:logistic", 
                # booster = "gblinear",
                eta                 = 0.02, #0.06, #0.01,
                max_depth           = 10,  # changed from default of 8
                subsample           = 0.5,
                colsample_bytree    = 0.5,
                min_child_weight    = 7,
                gamma               = 1,
                eval_metric         = "auc",
                # alpha = 0.0001, 
                lambda = 1
                )

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 800, #280, #125, #250, # changed from 300
                    verbose             = 2, 
                    early.stop.round    = NULL,
                    watchlist           = watchlist,
                    maximize            = TRUE)





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

