import time
#ddir = 'E:/workspace/data/cdiscount/'
#wdir = 'C:/Users/ngaude/Documents/GitHub/kaggle/cdiscount/'
ddir = '/home/ngaude/workspace/data/cdiscount/'
wdir = '/home/ngaude/workspace/github/kaggle/cdiscount/'

columns = ['Identifiant_Produit','Categorie1','Categorie2','Categorie3','Description','Libelle','Marque','Produit_Cdiscount','prix']
columns = {k:v for v,k in enumerate(columns)}

fname = ddir+'training_shuffled_normed.csv'
ffname = ddir+'training.vw'
ff = open(ffname,'w')

start_time = time.time()

counter=0
for i,line in enumerate(open(fname)):
    ls = line.split(';')
    row = "{0} {1}|M {2} |D {3} |L {4} |P price:{5}\n".format(
        int(ls[columns['Categorie3']])+1,
        ls[columns['Identifiant_Produit']],
        ls[columns['Marque']],
        ls[columns['Description']],
        ls[columns['Libelle']],
        ls[columns['prix']])
    ff.write(row)
    if i % 10000 == 0:
        print fname,i,'/15786885',int(time.time() - start_time),'s'
 
ff.close()

# vw training.vw -f model.vw --loss_function logistic -b 21 -P 1000 --oaa 5786
