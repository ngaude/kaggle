#!/usr/bin/python
import csv

with open('data/train.csv','rb') as f:
    reader = csv.reader(f, delimiter=',', quotechar='"')  
    header = next(reader,None)

with open('data/train.balanced.csv', 'rb') as finput, open('data/vw_train.dat','w') as foutput:
    reader = csv.reader(finput, delimiter=',', quotechar='"')  
    i = 1
    for row in reader:
        if len(row) != 1934:
            print i,row
            break
        i += 1
        if (i%1000)==0:
            print i
        if row[-1]=='0':
            label = '-1'
        else:
            label = '1'
        uid = row[0]
        features = [header[j]+'_'+row[j].translate(None,' []_,-:').replace('-','M').replace('.','P') for j in range(1,len(row)-1)]
        s = label + ' | '+ ' '.join(features)+'\n'
        foutput.write(s)


with open('data/test.csv', 'rb') as finput, open('data/vw_test.dat','w') as foutput:
    reader = csv.reader(finput, delimiter=',', quotechar='"')  
    header = next(reader,None)
    i = 1
    for row in reader:
        if len(row) != 1933:
            print i,row
            break
        i += 1
        if (i%1000)==0:
            print i
        label = '-1'
        uid = row[0]
        features = [header[j]+'_'+row[j].translate(None,' []_,-:').replace('-','M').replace('.','P') for j in range(1,len(row))]
        s = label + ' | '+ ' '.join(features)+'\n'
        foutput.write(s)

