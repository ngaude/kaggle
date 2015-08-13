import pandas as pd

ddir = '/home/ngaude/workspace/data/cdiscount/'

color_list = set('roug bleu vert jaun noir gris violet blanc orang marron ros beig'.split())

def get_txt_without_color(r):
    txt = (r.Marque+' '+r.Libelle+' '+r.Description).split()
    txt = ['adidi' if w in color_list else w for w in txt]
    return ' '.join(txt)

#######################
# textify the training 
# for searching purpose
#######################

df = pd.read_csv(ddir+'training_normed.csv',sep=';',names=['Identifiant_Produit','Categorie1','Categorie2','Categorie3','Description','Libelle','Marque','Produit_Cdiscount','prix']).fillna('')
df['uid'] = df.Categorie3.astype(str)+'@'+df.Identifiant_Produit.astype(str)
# while df will try to get the full one...
df['t_index'] = df.apply(get_txt_without_color,axis=1)
df = df[map(lambda t:len(t)>50,df.t_index)]
df = df[['t_index','uid']].sort(['t_index','uid']).reset_index(drop=True)
df.to_csv(ddir+'search_txt_full.tsv',sep='\t',index=False,header=False)

df = pd.read_csv(ddir+'training_normed.csv',sep=';',names=['Identifiant_Produit','Categorie1','Categorie2','Categorie3','Description','Libelle','Marque','Produit_Cdiscount','prix']).fillna('')
# questionning the Produit_Cdiscount quality, shall we take all :
# sdf focus on the cdiscount only products ....
sdf = df[df.Produit_Cdiscount == 1]
sdf['uid'] = sdf.Categorie3.astype(str)+'@'+sdf.Identifiant_Produit.astype(str)
sdf['t_index'] = sdf.apply(get_txt_without_color,axis=1)
sdf = sdf[map(lambda t:len(t)>50,sdf.t_index)]
sdf = sdf[['t_index','uid']].sort(['t_index','uid']).reset_index(drop=True)
sdf.to_csv(ddir+'search_txt.tsv',sep='\t',index=False,header=False)



#######################
# textify the test now
# for querying purpose
#######################

df = pd.read_csv(ddir+'test_normed.csv',sep=';',names=['Identifiant_Produit','Description','Libelle','Marque','prix']).fillna('')
df['t_index'] = df.apply(get_txt_without_color,axis=1)
df = df[map(lambda t:len(t)>50,df.t_index)]
df = df[['t_index','Identifiant_Produit']].sort(['t_index','Identifiant_Produit']).reset_index(drop=True)
df.to_csv(ddir+'query_txt.tsv',sep=';',index=False,header=False)



