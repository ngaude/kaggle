##################################
# http://code.google.com/p/apporo/
##################################

svn checkout http://tsubomi.googlecode.com/svn/trunk/ tsubomi
cd tsubomi/src/tsubomi
make
sudo make install
svn checkout http://apporo.googlecode.com/svn/trunk/ apporo
cd apporo/src
make
sudo make install


SEARCH_TXT=/home/ngaude/workspace/data/cdiscount/search_txt.tsv
split --lines=500000 -d $SEARCH_TXT `basename $SEARCH_TXT`

export LD_LIBRARY_PATH=/usr/local/lib/
SEARCH_TXT=/home/ngaude/workspace/data/cdiscount/search_txt.tsv

cd `dirname $SEARCH_TXT`

for i in {1..10}
do
    SEARCH_TXT_PART=`basename $SEARCH_TXT``printf "%02d" $i`
    apporo_indexer -i $SEARCH_TXT_PART -bt
    apporo_indexer -i $SEARCH_TXT_PART -d
done

for i in {11..21}
do
    SEARCH_TXT_PART=`basename $SEARCH_TXT``printf "%02d" $i`
    apporo_indexer -i $SEARCH_TXT_PART -bt
    apporo_indexer -i $SEARCH_TXT_PART -d
done

for i in {21..31}
do
    SEARCH_TXT_PART=`basename $SEARCH_TXT``printf "%02d" $i`
    apporo_indexer -i $SEARCH_TXT_PART -bt
    apporo_indexer -i $SEARCH_TXT_PART -d
done

cd -


