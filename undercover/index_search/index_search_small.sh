##################################
# http://code.google.com/p/apporo/
##################################

# FIRST PASS : QUERY from TEST and SEARCH to TRAIN : neighboring
SEARCH_TXT=/home/ngaude/workspace/data/cdiscount/search_txt.tsv

QUERY_TXT=/home/ngaude/workspace/data/cdiscount/query_txt.tsv
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:/lib

cd `dirname $SEARCH_TXT`

while read line; do
    echo '*****************************************'
    query_txt=`echo $line | cut -d';' -f1`
    query_id=`echo $line | cut -d';' -f2`
    echo 'query:'$query_id
    echo '*****:'`basename $SEARCH_TXT`,$query_txt
    apporo_searcher -i `basename $SEARCH_TXT` -s -p -r 7 -t 0.93 -q "$query_txt"
    echo '*****************************************'
done < $QUERY_TXT

cd -  > /dev/null
