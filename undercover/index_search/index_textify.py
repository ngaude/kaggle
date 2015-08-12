import pandas as pd

ddir = '/home/ngaude/workspace/data/cdiscount/'

df = pd.read_csv(ddir+'training_normed.csv',sep=';',names=['Identifiant_Produit','Categorie1','Categorie2','Categorie3','Description','Libelle','Marque','Produit_Cdiscount','prix']).fillna('')

# questionning the Produit_Cdiscount quality, shall we take all ?
#df = df[df.Produit_Cdiscount == 1]

df['t_index'] = df.Marque+' '+df.Libelle+' '+df.Description
df = df[['t_index','Categorie3']].sort('t_index').reset_index(drop=True)
df.to_csv(ddir+'search_txt.tsv',sep='\t',index=False,header=False)

df = pd.read_csv(ddir+'test_normed.csv',sep=';',names=['Identifiant_Produit','Description','Libelle','Marque','prix']).fillna('')
df['t_index'] = df.Marque+' '+df.Libelle+' '+df.Description
df = df[map(lambda t:len(t)>50,df.t_index)]
df = df[['t_index','Identifiant_Produit']].sort('t_index').reset_index(drop=True)
df.to_csv(ddir+'query_txt.tsv',sep=';',index=False,header=False)
