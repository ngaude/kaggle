#!/bin/sh

cd /home/ngaude/workspace/github/kaggle/springleaf/data
sed -e 1d train.csv | shuf > data.csv
head -n 1 train.csv > header.csv

cat header.csv > cv.csv
head -n 10000 data.csv >> cv.csv

cat header.csv > tr.csv
tail -n +100001 data.csv >> tr.csv

rm header.csv
rm data.csv
