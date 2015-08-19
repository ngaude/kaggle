
vw -b 21 --passes 10 --cache_file data/train.cache -f data/data.model -d data/vw_train.dat --loss_function logistic

vw -t --cache_file data/test.cache -i data/data.model -p vw_test.pred -d data/vw_test.dat


vw -b 21 --passes 10 --cache_file data/train.cache -f data/data.model -d data/vw_train.dat --loss_function logistic


vw -d data/vw_test.dat -t -i data/data.model  -r data/vw_test.raw_pred -p data/vw_test.pred --loss_function=logistic
vw -t --cache_file data/test.cache -i data/data.model -p vw_test.pred -d data/vw_test.dat
