rm data/train.cache data/data.model

vw -b 27 --passes 10 --cache_file data/train.cache -f data/data.model -d data/vw_train_weighted.dat --loss_function logistic --binary

vw -t -i data/data.model -d data/vw_test.dat --loss_function logistic -p data/vw_test.pred
