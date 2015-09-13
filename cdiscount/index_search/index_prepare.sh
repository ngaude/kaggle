##################################
# http://code.google.com/p/apporo/
##################################

# svn checkout http://tsubomi.googlecode.com/svn/trunk/ tsubomi
# cd tsubomi/src/tsubomi
# make
# sudo make install
# svn checkout http://apporo.googlecode.com/svn/trunk/ apporo
# cd apporo/src
# make
# sudo make install

export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib:/lib

# full scan NN search of 3M rows / 15M rows
#SEARCH_TXT=/home/ngaude/workspace/data/cdiscount/search_txt_full.tsv.3M

# precise scan NN search of the 574354 rows / 15M rows (cdiscount only)
SEARCH_TXT=/home/ngaude/workspace/data/cdiscount/search_txt.tsv


cd `dirname $SEARCH_TXT`

SEARCH_TXT_PART=`basename $SEARCH_TXT`
apporo_indexer -i $SEARCH_TXT_PART -bt
apporo_indexer -i $SEARCH_TXT_PART -d

cd -

